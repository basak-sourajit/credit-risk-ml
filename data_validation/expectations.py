import great_expectations as ge

def validate_dataframe(df):
    ge_df = ge.from_pandas(df)

    ge_df.expect_column_values_to_not_be_null("loan_amnt")
    ge_df.expect_column_values_to_be_between("annual_inc", min_value=0)
    ge_df.expect_column_values_to_be_between("dti", min_value=0, max_value=100)
    ge_df.expect_column_values_to_be_between("open_acc", min_value=0)

    return ge_df.validate()
