"""
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
"""

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!


class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    """One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    """

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        """Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        """
        random_state = self.random_state  # inherited RandomState object

        userids_tr_N, itemids_tr_N, ratings_tr_N = train_tuple

        N = len(userids_tr_N)

        assert N == len(itemids_tr_N) == len(ratings_tr_N)

        assert len(list(userids_tr_N)) > len(set(list(userids_tr_N)))

        ratings_per_user = ag_np.zeros((n_users, 2))
        ratings_per_mov = ag_np.zeros((n_items, 2))
        for i in range(N):
            ratings_per_user[userids_tr_N[i]] += ag_np.asarray([1, ratings_tr_N[i]])
            ratings_per_mov[itemids_tr_N[i]] += ag_np.asarray([1, ratings_tr_N[i]])

        # assert ag_np.any(ratings_per_mov[:, 0] == 0)
        avg_ratings_per_user = ag_np.asarray([x / n for n, x in ratings_per_user])
        avg_ratings_per_mov = ag_np.asarray(
            [x / n if n != 0 else 0 for n, x in ratings_per_mov]
        )

        assert ag_np.all(1 <= avg_ratings_per_user)
        assert ag_np.all(5 >= avg_ratings_per_user)
        # print(f"avg_ratings_per_mov   {avg_ratings_per_mov}")
        assert ag_np.all((0 == avg_ratings_per_mov) | (1 <= avg_ratings_per_mov))

        assert ag_np.all(5 >= avg_ratings_per_mov)

        assert len(avg_ratings_per_user) == n_users
        assert len(avg_ratings_per_mov) == n_items

        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            #
            mu=ag_np.asarray([ag_np.mean(ratings_tr_N)]),
            b_per_user=avg_ratings_per_user,
            c_per_item=avg_ratings_per_mov,
            # U=(ag_np.ones((n_users * self.n_factors))),
            # V=(ag_np.ones((n_items * self.n_factors))),
            U=(
                0.0001
                * random_state.randn(n_users * self.n_factors).reshape(
                    (n_users, self.n_factors)
                )
            ),
            V=(
                0.0001
                * random_state.randn(n_items * self.n_factors).reshape(
                    (n_items, self.n_factors)
                )
            ),
        )

    def predict(
        self,
        user_id_N,
        item_id_N,
        mu=None,
        b_per_user=None,
        c_per_item=None,
        U=None,
        V=None,
    ):
        """Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        """
        # We want ... mu + b_
        mu = self.param_dict["mu"] if mu is None else mu
        b_per_user = self.param_dict["b_per_user"] if b_per_user is None else b_per_user
        c_per_item = self.param_dict["c_per_item"] if c_per_item is None else c_per_item
        U = self.param_dict["U"] if U is None else U
        V = self.param_dict["V"] if V is None else V

        N = len(user_id_N)
        K = self.n_factors

        assert N == len(item_id_N)

        users_weights_NK, items_weights_NK = U[user_id_N], V[item_id_N]

        assert users_weights_NK.shape == items_weights_NK.shape
        inner_prod = users_weights_NK * items_weights_NK
        assert inner_prod.shape == (N, K)
        yhat_N = (
            mu
            + b_per_user[user_id_N]
            + c_per_item[item_id_N]
            + ag_np.sum(inner_prod, axis=1)
        )
        return yhat_N

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        """Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        """

        user_ids_B, item_ids_B, ratings_B = data_tuple

        B = len(user_ids_B)

        assert B == len(item_ids_B) == len(ratings_B)

        yhat_B = self.predict(user_ids_B, item_ids_B, **param_dict)
        total_error = ag_np.sum(ag_np.square(ratings_B - yhat_B))

        total_error += self.alpha * (
            ag_np.sum(ag_np.square(param_dict["U"]))
            + ag_np.sum(ag_np.square(param_dict["V"]))
        )

        return total_error


if __name__ == "__main__":
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = (
        load_train_valid_test_datasets()
    )
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1, n_factors=2, alpha=0.0
    )
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)

    model.predict(valid_tuple[0], valid_tuple[1])
