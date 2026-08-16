# BA-House PGExplainer Common Pool Freeze

## Methodological Status

This pool is for a post-hoc controlled sensitivity analysis. It does not
replace the original frozen Task 19 primary results.

## Construction

The common pool was defined by the symmetric intersection of the pre-existing
backbone-specific eligible pools using `node_id`.

The fact that the common pool equals the original GCN pool is an empirical
consequence of set inclusion and was not used as a manual selection rule.

## Counts

- GCN train: `48`
- GraphSAGE train: `106`
- Common train: `48`
- GCN validation: `16`
- GraphSAGE validation: `35`
- Common validation: `16`
- Final targets: `12`

## Files

- Common train: `configs\targets\ba_house_pg_common_train_targets.csv`
- Common train SHA256: `05e4574515fb7a03b846adb677295c940eb7af380216c15dfedf65bb9430bc3c`
- Common validation: `configs\targets\ba_house_pg_common_val_targets.csv`
- Common validation SHA256: `57fc6135495d9a6f8c7b8c1c3babcddd6b76187ebb353ed6fd8c289771980776`
- Final targets: `configs\targets\ba_house_2x2_targets.csv`
- Final targets SHA256: `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f`
