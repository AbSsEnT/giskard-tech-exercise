# ====Constants====
DATA_SOURCE_URL = "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/credit_scoring_classification_model_dataset/german_credit_prepared.csv"

GISKARD_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInRva2VuX3R5cGUiOiJBUEkiLCJhdXRoIjoiUk9MRV9BRE1JTiIsImV4cCI6MTY4ODA2MDkxMn0.a50g7gWXxLQaf8Jatqjw_UO1hJCLUXYJK_10wowv6mE"

COLUMN_TYPES = {
    'default': "category",
    'account_check_status': "category",
    'duration_in_month': "numeric",
    'credit_history': "category",
    'purpose': "category",
    'credit_amount': "numeric",
    'savings': "category",
    'present_employment_since': "category",
    'installment_as_income_perc': "numeric",
    'sex': "category",
    'personal_status': "category",
    'other_debtors': "category",
    'present_residence_since': "numeric",
    'property': "category",
    'age': "numeric",
    'other_installment_plans': "category",
    'housing': "category",
    'credits_this_bank': "numeric",
    'job': "category",
    'people_under_maintenance': "numeric",
    'telephone': "category",
    'foreign_worker': "category"
}

CONTINUOUS_FEATURES = [
    'credit_amount',
    'installment_as_income_perc'
]

DISCRETE_FEATURES = [
    'duration_in_month',
    'present_residence_since',
    'age',
    'credits_this_bank',
    'people_under_maintenance',
]

CATEGORICAL_FEATURES = [
    'account_check_status',
    'credit_history',
    'purpose',
    'savings',
    'present_employment_since',
    'sex',
    'personal_status',
    'other_debtors',
    'property',
    'other_installment_plans',
    'housing',
    'job',
    'telephone',
    'foreign_worker'
]
