#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
# from fanalysis.ca import CA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

# https://github.com/OlivierGarciaDev/fanalysis/blob/master/doc/ca_tutorial.ipynb

class Base(BaseEstimator, TransformerMixin):
    """ Base

    Don't use this class directly.
    Use the other classes of the package (CA, MCA, PCA), which inherit
    from this Base class.

    This Base class performs a Correspondence Analysis, given a
    contingency table containing absolute frequencies ;
    shape= n_rows x n_columns.

    This implementation only works for dense arrays.

    Parameters
    ----------
    n_components : int, float or None
        Number of components to keep.
        - If n_components is None, keep all the components.
        - If 0 <= n_components < 1, select the number of components such
          that the amount of variance that needs to be explained is
          greater than the percentage specified by n_components.
        - If 1 <= n_components :
            - If n_components is int, select a number of components
              equal to n_components.
            - If n_components is float, select the higher number of
              components lower than n_components.

    row_labels : array of strings or None
        - If row_labels is an array of strings : this array provides
          the row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.

    col_labels : array of strings or None
        - If col_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.

    stats : bool
        - If stats is true : stats are computed : contributions and
          square cosines for rows and columns.
        - If stats is false : stats are not computed.

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.

    row_labels_ : array of strings
        Labels for the rows.

    col_labels_ : array of strings
        Labels for the columns.

    col_labels_short_ : array of strings
        Short labels for the columns.
        Useful only for MCA, which inherits from Base class. In that
        case, the short labels for the columns at not prefixed by the
        names of the variables.

    eig_ : array of float
        A 3 x n_components_ matrix containing all the eigenvalues
        (1st row), the percentage of variance (2nd row) and the
        cumulative percentage of variance (3rd row).

    row_coord_ : array of float
        A n_rows x n_components_ matrix containing the row coordinates.

    col_coord_ : array of float
        A n_columns x n_components_ matrix containing the column
        coordinates.

    row_contrib_ : array of float
        A n_rows x n_components_ matrix containing the row
        contributions.

    col_contrib_ : array of float
        A n_columns x n_components_ matrix containing the column
        contributions.

    row_cos2_ : array of float
        A n_rows x n_components_ matrix containing the row cosines.

    col_cos2_ : array of float
        A n_columns x n_components_ matrix containing the column
        cosines.

    n_ : float
        The sum of the absolute frequencies in the X array.

    r_ : float
        The sum of the absolute frequencies in the X array, for each row
        (along axis = 1).
    c_ : float
        The sum of the absolute frequencies in the X array, for each
        column (along axis = 0).

    model_ : string
        The model fitted = 'base'
    """

    def __init__(self, n_components=None, row_labels=None, col_labels=None,
                 stats=True):
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.stats = stats

    def fit(self, X, y=None):
        """ Fit the model to X

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows in the number of rows and
            n_columns is the number of columns.
            X is a contingency table containing absolute frequencies.

        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Initializations
        self.n_ = np.sum(X)
        self.r_ = np.sum(X, axis=1).reshape(-1, 1)
        self.c_ = np.sum(X, axis=0).reshape(1, -1)

        # Stats initializations
        self.row_contrib_ = None
        self.col_contrib_ = None
        self.row_cos2_ = None
        self.col_cos2_ = None

        # Compute SVD
        self._compute_svd(X)

        # Set row labels
        nrows = X.shape[0]
        self.row_labels_ = self.row_labels
        if (self.row_labels_ is None) or (len(self.row_labels_) != nrows):
            self.row_labels_ = ["row" + str(x) for x in np.arange(0, nrows)]

        # Set column labels
        ncols = X.shape[1]
        self.col_labels_ = self.col_labels
        if (self.col_labels_ is None) or (len(self.col_labels_) != ncols):
            self.col_labels_ = ["col" + str(x) for x in np.arange(0, ncols)]
        self.col_labels_short_ = self.col_labels_
        self.model_ = "base"

        return self

    def transform(self, X, y=None):
        """ Apply the dimensionality reduction on X.

        X is projected on the first axes previous extracted from a
        training set.

        Parameters
        ----------
        X : array of float, shape (n_rows_sup, n_columns)
            New data, where n_rows_sup is the number of supplementary
            row points and n_columns is the number of columns.
            X rows correspond to supplementary row points that are
            projected on the axes.
            X is a contingency table containing absolute frequencies.

        y : None
            y is ignored.

        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points on the axes.
        """
        x = np.sum(X, axis=1).astype(float)
        return ((np.diag(x ** (-1)).dot(X)).dot((self.eig_[0]
                                                 ** (-0.5)).reshape(1, -1)
                                                * self.col_coord_))

    def _compute_svd(self, X):
        """ Compute a Singular Value Decomposition

        Then, this function computes :
            n_components_ : number of components.
            eig_ : eigen values.
            row_coord_ : row coordinates.
            col_coord_ : column coordinates.
            _compute_stats(X) : if stats is True.

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows is the number of rows and
            n_columns is the number of columns.
            X is a contingency table containing absolute frequencies.

        Returns
        -------
        None
        """
        # SVD
        E = (self.r_.dot(self.c_)) / self.n_
        R = (1.0 / np.sqrt(self.n_)) * (X - E) / np.sqrt(E)
        U, lambdas, V = np.linalg.svd(R, full_matrices=False)

        # Eigen values first step
        eigen_values = lambdas ** 2
        eigen_values_percent = 100 * eigen_values / np.sum(eigen_values)
        eigen_values_percent_cumsum = np.cumsum(eigen_values_percent)

        # Set n_components_
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = len(eigen_values) - 1
        elif (self.n_components_ >= 0) and (self.n_components_ < 1):
            i = 0
            threshold = 100 * self.n_components_
            while eigen_values_percent_cumsum[i] < threshold:
                i = i + 1
            self.n_components_ = i
        elif ((self.n_components_ >= 1)
              and (self.n_components_ <= len(eigen_values) - 1)
              and (isinstance(self.n_components_, int))):
            self.n_components_ = int(np.trunc(self.n_components_))
        elif ((self.n_components_ >= 1)
              and (self.n_components_ <= len(eigen_values) - 1)
              and (isinstance(self.n_components_, float))):
            self.n_components_ = int(np.floor(self.n_components_))
        else:
            self.n_components_ = len(eigen_values) - 1

        # 特征值 Eigen values second step
        self.eig_ = np.array([eigen_values[:self.n_components_],
                              eigen_values_percent[:self.n_components_],
                              eigen_values_percent_cumsum[:self.n_components_]])

        # Factor coordinates
        row_coord = (U / np.sqrt(self.r_ / self.n_)) * (lambdas.reshape(1, -1))
        col_coord = ((V.T / np.sqrt(self.c_.T / self.n_))
                     * (lambdas.reshape(1, -1)))
        self.row_coord_ = row_coord[:, :self.n_components_]
        self.col_coord_ = col_coord[:, :self.n_components_]

        # Compute stats
        if self.stats:
            self._compute_stats(X)

    def _compute_stats(self, X):
        """ Compute statistics :
                row_contrib_ : row contributions.
                col_contrib_ : column contributions.
                row_cos2_ : row cosines.
                col_cos2_ : column cosines.

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows is the number of rows and
            n_columns is the number of columns.
            X is a contingency table containing absolute frequencies.

        Returns
        -------
        None
        """
        # Contributions
        row_contrib = 100 * ((self.r_ / self.n_)
                             * (self.row_coord_ ** 2)
                             * (1 / self.eig_[0].T))
        col_contrib = 100 * ((self.c_.T / self.n_)
                             * (self.col_coord_ ** 2)
                             * (1 / self.eig_[0].T))
        self.row_contrib_ = row_contrib[:, :self.n_components_]
        self.col_contrib_ = col_contrib[:, :self.n_components_]

        # Cos2
        row_cos2 = ((self.row_coord_ ** 2)
                    / (np.sum((self.n_ / self.c_)
                              * ((X / self.r_) - (self.c_ / self.n_)) ** 2, axis=1)
                       .reshape(-1, 1)))
        col_cos2 = ((self.col_coord_ ** 2)
                    / (np.sum((self.n_ / self.r_)
                              * ((X / self.c_) - (self.r_ / self.n_)) ** 2, axis=0))
                    .reshape(-1, 1))
        self.row_cos2_ = row_cos2[:, :self.n_components_]
        self.col_cos2_ = col_cos2[:, :self.n_components_]

    def plot_eigenvalues(self, type="absolute", figsize=None):
        """ Plot the eigen values graph

        Parameters
        ----------
        type : string
            Select the graph to plot :
                - If "absolute" : plot the eigenvalues.
                - If "percentage" : plot the percentage of variance.
                - If "cumulative" : plot the cumulative percentage of
                  variance.
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        if type == "absolute":
            plt.bar(np.arange(1, self.eig_[0].shape[0] + 1), self.eig_[0],
                    color="steelblue", align="center")
            plt.xlabel("轴")  # Axis
            plt.ylabel("特征值")  # Eigenvalue
        elif type == "percentage":
            plt.bar(np.arange(1, self.eig_[1].shape[0] + 1), self.eig_[1],
                    color="steelblue", align="center")
            plt.xlabel("轴")  # Axis
            plt.ylabel("方差")  # "Percentage of variance"
        elif type == "cumulative":
            plt.bar(np.arange(1, self.eig_[2].shape[0] + 1), self.eig_[2],
                    color="steelblue", align="center")
            plt.xlabel("轴")  # Axis
            plt.ylabel("累积方差")  # Cumulative percentage of variance
        else:
            raise Exception("Error : 'type' variable must be 'absolute' or \
                            'percentage' or 'cumulative'")
        plt.title("陡坡图")  # Scree plot
        plt.show()

    def mapping(self, num_x_axis, num_y_axis, short_labels=True, figsize=None):
        """ Plot the Factor map for rows and columns simultaneously

        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.

        num_y_axis : int
             Select the component to plot as y axis.

        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.

        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        plt.scatter(self.row_coord_[:, num_x_axis - 1],
                    self.row_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        plt.scatter(self.col_coord_[:, num_x_axis - 1],
                    self.col_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        for i in np.arange(0, self.row_coord_.shape[0]):
            plt.text(self.row_coord_[i, num_x_axis - 1],
                     self.row_coord_[i, num_y_axis - 1],
                     self.row_labels_[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="red")
        for i in np.arange(0, self.col_coord_.shape[0]):
            plt.text(self.col_coord_[i, num_x_axis - 1],
                     self.col_coord_[i, num_y_axis - 1],
                     col_labels[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="blue")
        plt.title("因子图")  # Factor map
        plt.xlabel("维度 " + str(num_x_axis) + " ("
                   + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")  # Dim
        plt.ylabel("维度 " + str(num_y_axis) + " ("
                   + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        plt.show()

    def mapping_row(self, num_x_axis, num_y_axis, figsize=None):
        """ Plot the Factor map for rows only

        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.

        num_y_axis : int
             Select the component to plot as y axis.

        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        plt.scatter(self.row_coord_[:, num_x_axis - 1],
                    self.row_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        for i in np.arange(0, self.row_coord_.shape[0]):
            plt.text(self.row_coord_[i, num_x_axis - 1],
                     self.row_coord_[i, num_y_axis - 1],
                     self.row_labels_[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="red")
        plt.title("行因子图")  # Factor map for rows
        plt.xlabel("维度 " + str(num_x_axis) + " ("
                   + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")  # Dim
        plt.ylabel("维度 " + str(num_y_axis) + " ("
                   + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")  # Dim
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        plt.show()

    def mapping_col(self, num_x_axis, num_y_axis, short_labels=True,
                    figsize=None):
        """ Plot the Factor map for columns only

        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.

        num_y_axis : int
             Select the component to plot as y axis.

        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.

        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        plt.scatter(self.col_coord_[:, num_x_axis - 1],
                    self.col_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        for i in np.arange(0, self.col_coord_.shape[0]):
            plt.text(self.col_coord_[i, num_x_axis - 1],
                     self.col_coord_[i, num_y_axis - 1],
                     col_labels[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="blue")
        plt.title("列因子图")  # Factor map for columns
        plt.xlabel("维度 " + str(num_x_axis) + " ("
                   + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")  # Dim
        plt.ylabel("维度 " + str(num_y_axis) + " ("
                   + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        plt.show()

    def plot_row_contrib(self, num_axis, nb_values=None, figsize=None):
        """ Plot the row contributions graph

        For the selected axis, the graph represents the row
        contributions sorted in descending order.

        Parameters
        ----------
        num_axis : int
            Select the axis for which the row contributions are plotted.

        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.

        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_rows = len(self.row_labels_)
        n_labels = len(self.row_labels_)
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_rows - n_labels
        contribs = self.row_contrib_[:, num_axis - 1]
        contribs_sorted = np.sort(contribs)[limit:n_rows]
        labels = pd.Series(self.row_labels_)[np.argsort(contribs)] \
            [limit:n_rows]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, contribs_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("行贡献")  # Rows contributions
        plt.xlabel("贡献率 (%)")  # Contributions
        plt.ylabel("行")  # Rows
        plt.show()

    def plot_col_contrib(self, num_axis, nb_values=None, short_labels=True,
                         figsize=None):
        """ Plot the column contributions graph

        For the selected axis, the graph represents the column
        contributions sorted in descending order.

        Parameters
        ----------
        num_axis : int
            Select the axis for which the column contributions are
            plotted.

        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.

        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.

        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_cols = len(self.col_labels_)
        n_labels = len(self.col_labels_)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_cols - n_labels
        contribs = self.col_contrib_[:, num_axis - 1]
        contribs_sorted = np.sort(contribs)[limit:n_cols]
        labels = pd.Series(col_labels)[np.argsort(contribs)][limit:n_cols]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, contribs_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("列贡献")  # Columns contributions
        plt.xlabel("贡献率 (%)")  # Contributions
        plt.ylabel("列")  # Columns
        plt.show()

    def plot_row_cos2(self, num_axis, nb_values=None, figsize=None):
        """ Plot the row cosines graph

        For the selected axis, the graph represents the row cosines
        sorted in descending order.

        Parameters
        ----------
        num_axis : int
            Select the axis for which the row cosines are plotted.

        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.

        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_rows = len(self.row_labels_)
        n_labels = len(self.row_labels_)
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_rows - n_labels
        cos2 = self.row_cos2_[:, num_axis - 1]
        cos2_sorted = np.sort(cos2)[limit:n_rows]
        labels = pd.Series(self.row_labels_)[np.argsort(cos2)][limit:n_rows]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, cos2_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("行余弦")  # Rows cos2
        plt.xlabel("余弦")  # Cos2
        plt.ylabel("行")  # Rows
        plt.show()

    def plot_col_cos2(self, num_axis, nb_values=None, short_labels=True,
                      figsize=None):
        """ Plot the column cosines graph

        For the selected axis, the graph represents the column cosines
        sorted in descending order.

        Parameters
        ----------
        num_axis : int
            Select the axis for which the column cosines are plotted.

        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.

        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.

        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_cols = len(self.col_labels_)
        n_labels = len(self.col_labels_)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_cols - n_labels
        cos2 = self.col_cos2_[:, num_axis - 1]
        cos2_sorted = np.sort(cos2)[limit:n_cols]
        labels = pd.Series(col_labels)[np.argsort(cos2)][limit:n_cols]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, cos2_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("列余弦")  # Columns cos2
        plt.xlabel("余弦")  # Cos2
        plt.ylabel("列")  # Columns
        plt.show()

    def row_topandas(self):
        """ Export row data (row coordinates, row contributions,
        row cosines) into a pandas.DataFrame

        Parameters
        ----------
        None

        Returns
        -------
        Y : pandas.DataFrame, shape (n_rows, 3 * n_components_)
            - Column indices from 0 to (n_components_ - 1) :
              row coordinates.
            - Column indices from n_components_ to (2*n_components_-1) :
              row contributions.
            - Column indices from (2 * n_components_) to
              (3 * n_components_ - 1) : row cosines.
        """
        if self.stats:
            ndim = self.eig_[0].shape[0]
            nrows = self.row_coord_.shape[0]

            # Columns labels
            col = ["row_coord_dim" + str(x + 1) for x in np.arange(0, ndim)]
            col.extend(["row_contrib_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])
            col.extend(["row_cos2_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])

            # Row labels
            if ((self.row_labels_ is not None)
                    and (len(self.row_labels_) == nrows)):
                ind = self.row_labels_
            else:
                ind = ["row" + str(x) for x in np.arange(0, nrows)]

            return pd.DataFrame(np.c_[self.row_coord_,
                                      self.row_contrib_,
                                      self.row_cos2_],
                                index=ind, columns=col)

        else:
            raise Exception("Error : self.stats attribute set at \'False\'")

    def col_topandas(self):
        """ Export column data (column coordinates,
        column contributions, column cosines) into a pandas.DataFrame

        Parameters
        ----------
        None

        Returns
        -------
        Y : pandas.DataFrame, shape (n_columns, 3 * n_components_)
            - Column indices from 0 to (n_components_ - 1) :
              column coordinates
            - Column indices from n_components_ to (2*n_components_-1) :
              column contributions.
            - Column indices from (2 * n_components_) to
              (3 * n_components_ - 1) : column cosines.

        """
        if self.stats:
            ndim = self.eig_[0].shape[0]
            ncols = self.col_coord_.shape[0]

            # Columns labels
            col = ["col_coord_dim" + str(x + 1) for x in np.arange(0, ndim)]
            col.extend(["col_contrib_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])
            col.extend(["col_cos2_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])

            # Row labels

            if ((self.col_labels_ is not None)
                    and (len(self.col_labels_) == ncols)):
                ind = self.col_labels_
            else:
                ind = ["col" + str(x) for x in np.arange(0, ncols)]

            return pd.DataFrame(np.c_[self.col_coord_,
                                      self.col_contrib_,
                                      self.col_cos2_],
                                index=ind, columns=col)

        else:
            raise Exception("Error : self.stats attribute set at \'False\'")

class CA(Base):
    """ Correspondence Analysis (CA)

    This class inherits from the Base class.

    CA performs a Correspondence Analysis, given a contingency table
    containing absolute frequencies ; shape= n_rows x n_columns.

    This implementation only works for dense arrays.

    Parameters
    ----------
    n_components : int, float or None
        Number of components to keep.
        - If n_components is None, keep all the components.
        - If 0 <= n_components < 1, select the number of components such
          that the amount of variance that needs to be explained is
          greater than the percentage specified by n_components.
        - If 1 <= n_components :
            - If n_components is int, select a number of components
              equal to n_components.
            - If n_components is float, select the higher number of
              components lower than n_components.

    row_labels : array of strings or None
        - If row_labels is an array of strings : this array provides the
          row labels.
              If the shape of the array doesn't match with the number of
              rows : labels are automatically computed for each row.
        - If row_labels is None : labels are automatically computed for
          each row.

    col_labels : array of strings or None
        - If col_labels is an array of strings : this array provides the
          column labels.
              If the shape of the array doesn't match with the number of
              columns : labels are automatically computed for each
              column.
        - If col_labels is None : labels are automatically computed for
          each column.

    stats : bool
        - If stats is true : stats are computed : contributions and
          square cosines for rows and columns.
        - If stats is false : stats are not computed.

    Attributes
    ----------
    n_components_ : int
        The estimated number of components.

    row_labels_ : array of strings
        Labels for the rows.

    col_labels_ : array of strings
        Labels for the columns.

    col_labels_short_ : array of strings
        Short labels for the columns.
        Useful only for MCA, which inherits from Base class. In that
        case, the short labels for the columns at not prefixed by the
        names of the variables.

    eig_ : array of float
        A 3 x n_components_ matrix containing all the eigenvalues
        (1st row), the percentage of variance (2nd row) and the
        cumulative percentage of variance (3rd row).

    row_coord_ : array of float
        A n_rows x n_components_ matrix containing the row coordinates.

    col_coord_ : array of float
        A n_columns x n_components_ matrix containing the column
        coordinates.

    row_contrib_ : array of float
        A n_rows x n_components_ matrix containing the row
        contributions.

    col_contrib_ : array of float
        A n_columns x n_components_ matrix containing the column
        contributions.

    row_cos2_ : array of float
        A n_rows x n_components_ matrix containing the row cosines.

    col_cos2_ : array of float
        A n_columns x n_components_ matrix containing the column
        cosines.

    n_ : float
        The sum of the absolute frequencies in the X array.

    r_ : float
        The sum of the absolute frequencies in the X array, for each row
        (along axis = 1).
    c_ : float
        The sum of the absolute frequencies in the X array, for each
        column (along axis = 0).

    model_ : string
        The model fitted = 'ca'
    """

    def fit(self, X, y=None):
        """ Fit the model to X

        Parameters
        ----------
        X : array of float, shape (n_rows, n_columns)
            Training data, where n_rows in the number of rows and
            n_columns is the number of columns.
            X is a contingency table containing absolute frequencies.

        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        Base.fit(self, X, y=None)
        self.model_ = "ca"
        return self




def main():
    df = pd.DataFrame([[96, 118, 2, 71, 50, 17],
                     [122, 136, 11, 76, 49, 41],
                     [193, 184, 74, 63, 103, 79],
                     [360, 365, 63, 145, 141, 184],
                     [511, 593, 57, 217, 172, 306],
                     [385, 457, 42, 174, 104, 220],
                     [156, 185, 8, 69, 42, 85],
                     [1474, 1931, 181, 852, 642, 782]],
                      columns=[ '广播', '电话', '电报', '报纸', '短信', '电视'],
                      index= ['农民', '工程师', '学生', '律师',
           '教师', '医生', '司机', '售货员']
                      )

    # df = pd.DataFrame([[55, 349, 48, 146, 63],
    #                    [12, 159, 26, 62, 19],
    #                    [122, 439, 58, 178, 111]],
    #                   columns=['高中以下', '高中', '大专', '本科', '硕士'],
    #                   index=['老布什', '佩罗', '克林顿']
    #                   )

    X = df.as_matrix()
    my_ca = CA(row_labels=df.index.values, col_labels=df.columns.values)
    my_ca.fit(X)
    print(my_ca.eig_)
    my_ca.plot_eigenvalues()  # 默认是绝对值
    my_ca.plot_eigenvalues(type="percentage")  # 方差
    my_ca.plot_eigenvalues(type="cumulative")  # 累积方差
    df_rows = my_ca.row_topandas()

    print(df_rows)

    # Coordonnées des points lignes
    print(my_ca.row_coord_)

    # Contributions des points lignes
    print(my_ca.row_contrib_)

    # Cos2 des points lignes
    print(my_ca.row_cos2_)

    df_cols = my_ca.col_topandas()
    print(df_cols)

    # Coordonnées des points colonnes
    print(my_ca.col_coord_)

    # Contributions des points colonnes
    print(my_ca.col_contrib_)

    # Cos2 des points colonnes
    print(my_ca.col_cos2_)

    # Mapping simultané des points lignes et colonnes
    # Les paramètres de la méthode mapping indiquent que ce sont les axes 1 et 2 qui sont ici représentés
    my_ca.mapping(num_x_axis=1, num_y_axis=2)

    # Mapping des points lignes
    my_ca.mapping_row(num_x_axis=1, num_y_axis=2)

    # Mapping des points colonnes
    my_ca.mapping_col(num_x_axis=1, num_y_axis=2)

    # Classement des points lignes en fonction de leur contribution au 1er axe
    # Le paramètre de la méthode plot_row_contrib indique que c'est pour l'axe numéro 1 que les contributions sont ici
    # représentées
    my_ca.plot_row_contrib(num_axis=1)

    # Classement des points lignes en fonction de leur cos2 sur le 1er axe
    my_ca.plot_row_cos2(num_axis=1)

    # Classement des points colonnes en fonction de leur contribution au 1er axe
    my_ca.plot_col_contrib(num_axis=1)

    # Classement des points colonnes en fonction de leur cos2 sur le 1er axe
    my_ca.plot_col_cos2(num_axis=1)

    my_ca.mapping(1, 2, figsize=(10, 8))

    my_ca = CA(stats=False)

    # Estimation du modèle
    my_ca.fit(X)

    # Projection de l'ensemble des points lignes sur les axes factoriels
    my_ca.transform(X)

    # Idem en une seule étape
    my_ca.fit_transform(X)

    # Projection d'une ligne supplémentaire
    my_ca.transform([[25, 54, 41, 11, 65, 44]])


if __name__ == '__main__':
    main()