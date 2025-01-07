from .policy import BaseTextPolicy, HiddenVariableTextPolicy, FullSysTextPolicy, CoTTextPolicy

POLICIES = {
"base": BaseTextPolicy,
"hidden-sys": HiddenVariableTextPolicy,
"full-sys": FullSysTextPolicy,
"cot": CoTTextPolicy
}