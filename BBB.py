import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)

def gaussian(x, mu, sigma):
    bell = tf.exp(- (x - mu)**2 / (2.0 * sigma**2))
    return tf.clip_by_value(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)

def scale_mixture_prior(input, PI, SIGMA_1, SIGMA_2):
    prob1 = PI * gaussian(input, 0., SIGMA_1)
    prob2 = (1. - PI) * gaussian(input, 0., SIGMA_2)
    return tf.math.log(prob1 + prob2)

class BayesianDense(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, PI, SIGMA_1, SIGMA_2, google_init=False, scalar_mixture_prior=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.PI = PI
        self.SIGMA_1 = SIGMA_1
        self.SIGMA_2 = SIGMA_2
        self.scalar_mixture_prior = scalar_mixture_prior

        # Initialization
        mu_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05 if google_init else 0.1)
        rho_init = tf.keras.initializers.RandomNormal(mean=-5.0, stddev=0.05) if google_init else tf.keras.initializers.RandomUniform(minval=-3., maxval=-3.)

        self.weight_mu = self.add_weight(name="weight_mu", shape=(in_features, out_features), initializer=mu_init, trainable=True)
        self.weight_rho = self.add_weight(name="weight_rho", shape=(in_features, out_features), initializer=rho_init, trainable=True)
        self.bias_mu = self.add_weight(name="bias_mu", shape=(out_features,), initializer=mu_init, trainable=True)
        self.bias_rho = self.add_weight(name="bias_rho", shape=(out_features,), initializer=rho_init, trainable=True)

        self.lpw = 0.
        self.lqw = 0.

    def call(self, inputs, training=False):
        if not training:
            return tf.matmul(inputs, self.weight_mu) + self.bias_mu

        weight_sigma = tf.math.softplus(self.weight_rho)
        bias_sigma = tf.math.softplus(self.bias_rho)

        eps_w = tf.random.normal(shape=self.weight_mu.shape)
        eps_b = tf.random.normal(shape=self.bias_mu.shape)
        weight = self.weight_mu + weight_sigma * eps_w
        bias = self.bias_mu + bias_sigma * eps_b

        if self.scalar_mixture_prior:
            lpw = tf.reduce_sum(scale_mixture_prior(weight, self.PI, self.SIGMA_1, self.SIGMA_2))
            lpw += tf.reduce_sum(scale_mixture_prior(bias, self.PI, self.SIGMA_1, self.SIGMA_2))
        else:
            lpw = tf.reduce_sum(tf.math.log(gaussian(weight, 0, self.SIGMA_1))) + \
                  tf.reduce_sum(tf.math.log(gaussian(bias, 0, self.SIGMA_1)))

        lqw = tf.reduce_sum(tf.math.log(gaussian(weight, self.weight_mu, weight_sigma))) + \
              tf.reduce_sum(tf.math.log(gaussian(bias, self.bias_mu, bias_sigma)))

        self.lpw = lpw
        self.lqw = lqw
        return tf.matmul(inputs, weight) + bias
    
class BayesianRegressionNetwork(tf.keras.Model):
    def __init__(self, input_dim, hidden_layers, output_dim, samples, num_batches,
                 pi, sigma1, sigma2, google_init=False, scalar_mixture_prior=True):
        super().__init__()
        self.samples = samples
        self.num_batches = num_batches
        self.bbb_layers = []

        prev_dim = input_dim
        for dim in hidden_layers:
            self.bbb_layers.append(
                (BayesianDense(prev_dim, dim, pi, sigma1, sigma2, google_init, scalar_mixture_prior), tf.nn.relu)
            )
            prev_dim = dim

        self.output_layer = BayesianDense(prev_dim, output_dim, pi, sigma1, sigma2, google_init, scalar_mixture_prior)

    def call(self, x, training=False):
        for layer, act_fn in self.bbb_layers:
            x = act_fn(layer(x, training=training))
        return self.output_layer(x, training=training)

    def compute_bbb_loss(self, x, y, batch_idx=None):
        total_lpw, total_lqw, total_log_likelihood = 0., 0., 0.

        for _ in range(self.samples):
            y_pred = self(x, training=True)
            lpw = sum(layer.lpw for layer, _ in self.bbb_layers) + self.output_layer.lpw
            lqw = sum(layer.lqw for layer, _ in self.bbb_layers) + self.output_layer.lqw
            log_likelihood = -0.5 * tf.reduce_sum((y - y_pred) ** 2)

            total_lpw += lpw
            total_lqw += lqw
            total_log_likelihood += log_likelihood

        avg_lpw = total_lpw / self.samples
        avg_lqw = total_lqw / self.samples
        avg_log_likelihood = total_log_likelihood / self.samples

        if batch_idx is None:
            return (1. / self.num_batches) * (avg_lqw - avg_lpw) - avg_log_likelihood
        else:
            weight = 2. ** (self.num_batches - batch_idx - 1.) / (2. ** self.num_batches - 1)
            return weight * (avg_lqw - avg_lpw) - avg_log_likelihood
    
    
    def fit_bbb(self, inputs_train, outputs_train, inputs_val, outputs_val,
                optimizer, num_features, epochs=2000, use_early_stopping=False, patience=100, verbose=True, start_log_epoch=70):

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            # ---- Training ----
            with tf.GradientTape() as tape:
                train_loss = self.compute_bbb_loss(inputs_train[:, :num_features], outputs_train)
            grads = tape.gradient(train_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            # ---- Validation ----
            val_loss = self.compute_bbb_loss(inputs_val[:, :num_features], outputs_val)

            train_losses.append(train_loss.numpy())
            val_losses.append(val_loss.numpy())

            # ---- Early stopping ----
            if use_early_stopping:
                if val_loss.numpy() < best_val_loss:
                    best_val_loss = val_loss.numpy()
                    patience_counter = 0
                    best_weights = [w.numpy() for w in self.trainable_variables]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1} - Best Val Loss: {best_val_loss:.4f}")
                        break

            # ---- Logging ----
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.numpy():.4f} - Val Loss: {val_loss.numpy():.4f}")

        # ---- Restore best weights if early stopping used ----
        if use_early_stopping and best_weights is not None:
            for var, best in zip(self.trainable_variables, best_weights):
                var.assign(best)

        self.train_losses = train_losses
        self.val_losses = val_losses
        
        return train_losses, val_losses

    def plot_losses(self, start_log_epoch=70):
        # ---- Plotting ----
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_losses[start_log_epoch:], label='Train Loss')
        plt.plot(self.val_losses[start_log_epoch:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Negative ELBO Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')
        plt.grid(True)
        plt.show()

