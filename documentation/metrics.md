# Metric recorded

Kerastuner output two set of metric:

1. Execution metrics: metrics that are computed for each architecture execution.
2. Instance summary: metrics for an instance of the archicture. Include all executions metrics and metrics at aggregate at the instance level

By design KerasTuner do not output aggregate metrics over multiple instances as they are meaningless while doing horizontal scaling. Instead those aggregate metrics are left to either Keraslyzer (local) or Keraslyzer-service which perform them in the cloud.

