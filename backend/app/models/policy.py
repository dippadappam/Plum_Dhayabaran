"""Policy loader. The policy rulebook is read from policy_terms.json at
runtime and validated into typed models. No policy value is hardcoded
anywhere else in the system.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FamilyFloater(BaseModel):
    enabled: bool
    combined_limit: float
    covered_relationships: list[str]


class Coverage(BaseModel):
    sum_insured_per_employee: float
    annual_opd_limit: float
    per_claim_limit: float
    family_floater: FamilyFloater


class CategoryTerms(BaseModel):
    sub_limit: float
    copay_percent: float = 0
    network_discount_percent: float = 0
    branded_drug_copay_percent: Optional[float] = None
    generic_mandatory: Optional[bool] = None
    requires_prescription: bool = False
    requires_pre_auth: bool = False
    pre_auth_threshold: Optional[float] = None
    high_value_tests_requiring_pre_auth: list[str] = Field(default_factory=list)
    requires_dental_report: Optional[bool] = None
    requires_registered_practitioner: Optional[bool] = None
    max_sessions_per_year: Optional[int] = None
    covered: bool = True
    covered_procedures: list[str] = Field(default_factory=list)
    excluded_procedures: list[str] = Field(default_factory=list)
    covered_items: list[str] = Field(default_factory=list)
    excluded_items: list[str] = Field(default_factory=list)
    covered_systems: list[str] = Field(default_factory=list)


class WaitingPeriods(BaseModel):
    initial_waiting_period_days: int
    pre_existing_conditions_days: int
    specific_conditions: dict[str, int]


class Exclusions(BaseModel):
    conditions: list[str]
    dental_exclusions: list[str]
    vision_exclusions: list[str]


class PreAuthorization(BaseModel):
    required_for: list[str]
    validity_days: int


class SubmissionRules(BaseModel):
    deadline_days_from_treatment: int
    minimum_claim_amount: float
    currency: str


class FraudThresholds(BaseModel):
    same_day_claims_limit: int
    monthly_claims_limit: int
    high_value_claim_threshold: float
    auto_manual_review_above: float
    fraud_score_manual_review_threshold: float


class Member(BaseModel):
    member_id: str
    name: str
    date_of_birth: date
    gender: str
    relationship: str
    join_date: Optional[date] = None
    dependents: list[str] = Field(default_factory=list)
    primary_member_id: Optional[str] = None


class PolicyHolder(BaseModel):
    company_name: str
    employee_count: int
    policy_start_date: date
    policy_end_date: date
    renewal_status: str


class Policy(BaseModel):
    policy_id: str
    policy_name: str
    insurer: str
    policy_holder: PolicyHolder
    coverage: Coverage
    opd_categories: dict[str, CategoryTerms]
    waiting_periods: WaitingPeriods
    exclusions: Exclusions
    pre_authorization: PreAuthorization
    network_hospitals: list[str]
    submission_rules: SubmissionRules
    document_requirements: dict[str, dict[str, list[str]]]
    fraud_thresholds: FraudThresholds
    members: list[Member]

    def get_member(self, member_id: str) -> Optional[Member]:
        return next((m for m in self.members if m.member_id == member_id), None)

    def category_terms(self, category: str) -> Optional[CategoryTerms]:
        return self.opd_categories.get(category.lower())

    def member_join_date(self, member: Member) -> Optional[date]:
        """Dependents inherit the primary member's join date."""
        if member.join_date:
            return member.join_date
        if member.primary_member_id:
            primary = self.get_member(member.primary_member_id)
            if primary:
                return primary.join_date
        return None

    def is_network_hospital(self, hospital_name: Optional[str]) -> bool:
        """True when the bill's hospital name contains ALL the significant words
        of a network hospital's name (token-subset). A branch/address suffix
        ('Apollo Hospitals, Bannerghatta Road') still matches 'Apollo
        Hospitals', while a short partial token ('Max', 'Apollo Pharmacy') does
        not over-match a network entry."""
        if not hospital_name:
            return False
        bill_tokens = set(re.findall(r"[a-z0-9]+", hospital_name.lower()))
        if not bill_tokens:
            return False
        for h in self.network_hospitals:
            net_tokens = set(re.findall(r"[a-z0-9]+", h.lower()))
            if net_tokens and net_tokens <= bill_tokens:
                return True
        return False


_DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "data" / "policy_terms.json"


def load_policy(path: Optional[Path] = None) -> Policy:
    """Load and validate the policy rulebook from JSON."""
    policy_path = path or _DEFAULT_POLICY_PATH
    with open(policy_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return Policy.model_validate(raw)
