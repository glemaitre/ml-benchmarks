# Notes regarding the profiling of xgboost

## Profiler C/C++ code in Python

Requirements:

* `gperftools` C++ package
* `yep` Python package

### `gperftools`

In Ubuntu 14.04 LTS, there is an issue with the current version (i.e., 2.2.1).
There is a need to install an earlier version containing the following [PR](https://github.com/gperftools/gperftools/pull/779).
Otherwise, there is missing information regarding function call as presented in this [issue](https://github.com/gperftools/gperftools/issues/752).
You can find the dev version on [GitHub](https://github.com/gperftools/gperftools).
