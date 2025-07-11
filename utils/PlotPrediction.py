import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def prediction(model, SoH_RUL, input_test, num_features, output_test, num_path=50, threshold=2):

    preds = [model(input_test[:,:num_features], training=True) for _ in range(num_path)]
    preds_stack = tf.stack(preds, axis=0)

    mean_pred = tf.reduce_mean(preds_stack, axis=0).numpy().flatten()/output_test[0] if SoH_RUL == 'SoH' else tf.reduce_mean(preds_stack, axis=0).numpy().flatten()
    std_pred = tf.math.reduce_std(preds_stack, axis=0).numpy().flatten()/output_test[0] if SoH_RUL == 'SoH' else tf.math.reduce_std(preds_stack, axis=0).numpy().flatten()

    # Ground truth
    y_true = tf.reshape(output_test, [-1]).numpy()
    y_true = y_true/y_true[0] if SoH_RUL == 'SoH' else y_true
    # BNN prediction (mean)
    y_pred = mean_pred.numpy() if SoH_RUL == 'SoH' else mean_pred

    # Residuals
    residuals = y_true - y_pred

    # Compute Z-score of residuals
    z_scores = zscore(residuals)

    # Threshold (keep points where |z| < Threshold)
    mask = np.abs(z_scores) < threshold

    # Filtered values
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    cycle_filtered = input_test[:, num_features].numpy().flatten()[mask]

    masked_mean_pred = mean_pred[mask]
    masked_std_pred = std_pred[mask]

    return cycle_filtered, y_true_filtered, y_pred_filtered, masked_mean_pred, masked_std_pred

def Plot_predictions (model, inputs_test, output_test, Cycle_list,
                        Capacity_list, RUL_list,  SoH_RUL, num_features,
                        train_conditions, test_conditions, num_path = 50):

    cycle_filtered, y_true_filtered, y_pred_filtered, mean_pred, std_pred = prediction(model=model,
                                                                                        SoH_RUL=SoH_RUL, input_test=inputs_test,
                                                                                        output_test=output_test,
                                                                                        num_features=num_features, num_path=num_path, threshold=2)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.8, 4))

    plot_list = Capacity_list if SoH_RUL == 'SoH' else RUL_list

    for i, plot_path in enumerate(plot_list):
        path = plot_path/plot_path[0] if SoH_RUL == 'SoH' else plot_path
        plt.plot(Cycle_list[i], path, label=f'Cell {train_conditions[i]}', color=colors[i])

    plt.plot(cycle_filtered, y_true_filtered, 'x', label=f"Test data (Cell {test_conditions[0]})", lw = 2)
    plt.plot(cycle_filtered, y_pred_filtered, '.r', label="Prediction", lw = 5)

    plt.fill_between(
        cycle_filtered,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        alpha=0.3,
        label='Uncertainty (±2 std)')


    plt.xlabel("Cycle Index", ).set_size(15)
    plt.ylabel(SoH_RUL).set_size(15)
    plt.title(f"Bayesian Neural Network {SoH_RUL} Estimation").set_size(15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid('on')
    plt.show()