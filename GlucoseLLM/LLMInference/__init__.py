from .policy import BaseTextPolicy, HiddenVariableTextPolicy, FullSysTextPolicy, CoTTextPolicy, MajorityVotingTextPolicy

POLICIES = {
"base": BaseTextPolicy,
"hidden-sys": HiddenVariableTextPolicy,
"full-sys": FullSysTextPolicy,
"cot": CoTTextPolicy,
"majority-voting": MajorityVotingTextPolicy
}