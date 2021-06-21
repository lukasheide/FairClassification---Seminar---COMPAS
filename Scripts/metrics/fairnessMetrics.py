from sklearn.metrics import confusion_matrix
import pandas as pd


class FairnessMetrics():
    def __init__(self, y_pred, y_actual, x_test, sensitive_attr):
        self.y_pred = y_pred
        self.y_actual = y_actual
        self.x_test = x_test
        self.sensitive_attr = sensitive_attr

        # Extract indices of protected and unprotected class:
        self.indices_protected = x_test.index[x_test[sensitive_attr] == 0].tolist()
        self.indices_unprotected = x_test.index[x_test[sensitive_attr] == 1].tolist()

        # Count number of elements in both classes:
        self.n_prot = len(self.indices_protected)
        self.n_unprot = len(self.indices_unprotected)

        # Split up y_pred and actual based on sensitive attribute:
        self.y_pred_protected = y_pred.loc[self.indices_protected]
        self.y_pred_unprotected = y_pred.loc[self.indices_unprotected]

        self.y_actual_protected = y_actual.loc[self.indices_protected]
        self.y_actual_unprotected = y_actual.loc[self.indices_unprotected]

        ### Calculate confusion matrix for both classes:
        self.cm_prot = confusion_matrix(self.y_actual_protected, self.y_pred_protected)
        self.cm_unprot = confusion_matrix(self.y_actual_unprotected, self.y_pred_unprotected)

        # Get TN, FN, TP and FP for both classes:
        self.TN_prot = self.cm_prot[0][0]
        self.FN_prot = self.cm_prot[1][0]
        self.TP_prot = self.cm_prot[1][1]
        self.FP_prot = self.cm_prot[0][1]

        self.TN_unprot = self.cm_unprot[0][0]
        self.FN_unprot = self.cm_unprot[1][0]
        self.TP_unprot = self.cm_unprot[1][1]
        self.FP_unprot = self.cm_unprot[0][1]

        ##### Fairness Metrics #####

    def disparate_impact(self):
        ## Disparate Impact
        # Probability of positive prediction given unprotected/privileged class:
        # P(Y_hat = 1 | S = 1) = P(Y_hat = 1 | Y=1, S = 1) + P(Y_hat = 1 | Y=0, S = 1) = (TP + FP | S = 1) / N_S=1
        self.prob_pos_pred_given_prot = (self.TP_prot + self.FP_prot) / self.n_prot
        self.prob_pos_pred_given_unprot = (self.TP_unprot + self.FP_unprot) / self.n_unprot

        self.disparate_impact = self.prob_pos_pred_given_prot / self.prob_pos_pred_given_unprot

        return self.disparate_impact

    def true_positive_rate_balance(self):
        # P(Y_hat = 1 | Y=1, S = 1) - P(Y_hat = 1 | Y=1, S = 0)
        true_positive_rate_balance = (self.TP_unprot / self.n_unprot) - (self.TP_prot / self.n_prot)
        return true_positive_rate_balance

    def true_negative_rate_balance(self):
        # P(Y_hat = 0 | Y=0, S = 1) - P(Y_hat = 0 | Y=0, S = 0)
        true_negative_rate_balance = (self.TN_unprot / self.n_unprot) - (self.TN_prot / self.n_prot)
        return true_negative_rate_balance


class CausalDiscrimination():
    def __init__(self, y_pred, y_test, x_test, sensitive_attr, model):
        self.y_pred = y_pred
        self.y_actual = y_test
        self.x_test = x_test
        self.sensitive_attr = sensitive_attr

        self.model = model

    def compute_causal_discrimination(self):
        """
        Computes changes in y_pred, directly caused by changing the sensitive-attribute in the test-set
        and reapplying the model.
        :return:
        """
        # Create new test set:
        x_test_swapped_sensitive_attr = self.x_test.copy()

        # Swap values of sensitive attribute in x_test:
        x_test_swapped_sensitive_attr[self.sensitive_attr] = 1 - x_test_swapped_sensitive_attr[self.sensitive_attr]

        # Get new predictions:
        self.new_predictions = pd.DataFrame(self.model.predict(x_test_swapped_sensitive_attr),
                                            index=x_test_swapped_sensitive_attr.index,
                                            columns=['y_pred_new'])

        # Compute differences in predicted labels:
        diff = self.new_predictions['y_pred_new'] - self.y_pred['y_pred_test']

        # Count number of rows for which a change in the sensitive attribute caused a change in the predictions
        num_diff = abs(diff).sum()

        # Divide absolute number of changes by total number of elements:
        return num_diff/len(x_test_swapped_sensitive_attr)
