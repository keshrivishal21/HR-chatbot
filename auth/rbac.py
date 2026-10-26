BASE_NAMESPACE = ["general_docs", "general_faqs"]

ALL_NAMESPACES = [
    "general_docs",
    "general_faqs",
    "hr_policies",
    "engineering_docs",
    "leadership_docs",
]

ALL_ROLES = ["Admin", "Manager", "HR", "Engineer"]

ROLE_NAMESPACES: dict[str, list[str]] = {
    "Admin": ["hr_policies","engineering_docs","leadership_docs",],
    "Manager": [ "hr_policies", "engineering_docs"],
    "HR": [ "hr_policies"],
    "Engineer": ["engineering_docs"],
}


def get_accessible_namespaces(role: str) -> list[str]:
    role_namespaces = ROLE_NAMESPACES.get(role, [])
    merged = BASE_NAMESPACE + role_namespaces
    
    return merged


def can_upload(role: str) -> bool:
    return role == "Admin"


def can_manage_users(role: str) -> bool:
    return role == "Admin"
