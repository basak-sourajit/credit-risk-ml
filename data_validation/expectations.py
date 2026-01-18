from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig, InMemoryStoreBackendDefaults
from great_expectations.core.batch import RuntimeBatchRequest

def validate_dataframe(df):
    # ----------------------------
    # 1️⃣ Create DataContext
    # ----------------------------
    project_config = DataContextConfig(
        datasources={
            "default_pandas_datasource": {
                "class_name": "Datasource",
                "execution_engine": {"class_name": "PandasExecutionEngine"},
                "data_connectors": {
                    "default_runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"]
                    }
                }
            }
        },
        store_backend_defaults=InMemoryStoreBackendDefaults()
    )

    context = BaseDataContext(project_config=project_config)

    # ----------------------------
    # 2️⃣ Create a Validator for the DataFrame
    # ----------------------------
    batch_request = RuntimeBatchRequest(
        datasource_name="default_pandas_datasource",
        data_connector_name="default_runtime_data_connector",
        data_asset_name="loan_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_batch"},
    )

    validator = context.get_validator(batch_request=batch_request)

    # ----------------------------
    # 3️⃣ Run expectations
    # ----------------------------
    # Example: just validate existing expectations (or you can define inline)
    results = validator.validate()  

    return results
