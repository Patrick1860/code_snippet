from sklearn.grid_search import GridSearchCV


class NGridSearchCV(object):

    """
    A simple wrappers of GridSearchCV that can select the best model and parameters
    Important members are fit, predict.

    Note: Demo only. It's no longer needed as Sklearn has incorporated similar features in newer versions

    NGridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    and "transform" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Parameters
    ----------
    estimators : a dictionary of estimator objects.
        Estimators dictionary with estimator names (string) as keys and estimator objects as value
        Estimators need to be comparable such as they are all classification or regression model
    param_grids : a dictionary of parameters dictionary
        Parameters dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values.
    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Examples:
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    data = (X, y)

    dt = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    svm = SVC()
    estimators = {'DecisionTree': dt,
                  'RandomForestClassifier': rfc,
                  'SVM': svm}

    param_grids = {'DecisionTree': {'max_depth': [3, 10]},
                   'RandomForestClassifier': {'n_estimators': [10, 100]},
                   'SVM': {'C': [0.1, 0.5]}
                   }

    model = NGridSearchCV(estimators=estimators, param_grids=param_grids, verbose=1)
    model.fit(X, y)
    """

    def __init__(self, estimators, param_grids, scoring=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):

        self.estimators = estimators
        self.param_grids = param_grids
        self.GridSearchCV_collection = {}

        self.best_estimator_name_ = None
        self.best_gridsearch_cv = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

        for model_name, estimator in estimators.items():
            # initialize a GridSearchCV object for each estimator
            parameters = param_grids[model_name]
            grid_search = GridSearchCV(estimator,
                                       parameters,
                                       scoring=scoring,
                                       n_jobs=n_jobs,
                                       iid=iid,
                                       refit=refit,
                                       cv=cv,
                                       verbose=verbose,
                                       pre_dispatch=pre_dispatch,
                                       error_score=error_score)

            self.GridSearchCV_collection[model_name] = grid_search

    def fit(self, X, y=None):
        """Run fit with all models and all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        """
        model_scores = {}
        for model_name, grid_search in self.GridSearchCV_collection.items():
            grid_search.fit(X, y)
            model_scores[model_name] = grid_search.best_score_

        # get the best score and determine the best GridSearchCV
        self.best_estimator_name_ = max(model_scores, key=model_scores.get)
        self.best_gridsearch_cv = self.GridSearchCV_collection[self.best_estimator_name_]

        self.best_estimator_ = self.best_gridsearch_cv.best_estimator_
        self.best_params_ = self.best_gridsearch_cv.best_params_
        self.best_score_ = self.best_gridsearch_cv.best_score_

    def predict(self, X):
        """Call predict on the estimator with the best model and best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best model and best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best model and best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self.best_estimator_.predict_log_proba(X)

    def decision_function(self, X):
        """Call decision_function on the estimator with the best model and best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        return self.best_estimator_.decision_function(X)

    def transform(self, X):
        """Call transform on the estimator with the best model and best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        return self.best_estimator_.transform(X)
