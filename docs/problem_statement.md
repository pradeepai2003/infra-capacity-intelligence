# Problem Statement

Psiog Digital's platform teams manage infrastructure that supports data
pipelines, analytics platforms, and enterprise applications. Currently,
infrastructure capacity planning is performed using assumptions, historical
experience, and manually added safety buffers instead of real utilization
analytics and predictive forecasting.

**Consequences of the current approach:**
- Compute clusters operating at ~20% utilization continue running at full
  capacity because there is no evidence-based right-sizing process.
- Storage systems growing steadily month-over-month reach critical capacity
  limits unexpectedly because growth trends are not continuously tracked or
  forecasted.
- Infrastructure provisioned for completed projects often remains active for
  long periods due to the absence of utilization-based review mechanisms,
  creating orphaned resources and unnecessary operational costs.
- Genuine capacity shortages are discovered only after failures occur -- ETL
  jobs failing from insufficient compute headroom, storage I/O bottlenecks
  slowing pipelines, or analytics workloads degrading during seasonal spikes.

## Goal

Build an intelligent infrastructure capacity data system that continuously
collects utilization metrics, forecasts future demand with ML, proactively
detects capacity risks, and generates actionable right-sizing recommendations
before infrastructure waste or shortfalls become business problems.

See `architecture.md` and `module_design.md` for how this repository
implements that goal.
