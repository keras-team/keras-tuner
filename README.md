

## Cloud service
Make sure to enable default account: `gcloud auth application-default login`
Otherwise you will get an error like:

```
2018-06-27 10:00:15.891357: I tensorflow/core/platform/cloud/retrying_utils.cc:77] The operation failed and will be automatically retried in 1.06535 seconds (attempt 1 out of 10), caused by: Unavailable: Error executing an HTTP request (error code 6, error message 'Couldn't resolve host name')
```

## Checkout 

```git fetch <remote> <rbranch>:<lbranch> 
git checkout <lbranch>```