from threadpoolctl import threadpool_limits 


threadpool_limits(limits=1, user_api='blas')
threadpool_limits(limits=1, user_api='openmp')

