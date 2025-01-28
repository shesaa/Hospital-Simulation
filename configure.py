from base import Capacity, SectionType
from typing import List, Type, Optional, Dict


SECTION_CAPACITIES: Dict[SectionType, Capacity] = {
    SectionType.OUTSIDE: Capacity(servers=1, queue=None),
    SectionType.PRE_SURGERY: Capacity(servers=25, queue=None),
    SectionType.EMERGENCY: Capacity(servers=10, queue=10),
    SectionType.LABRATORY: Capacity(servers=3, queue=None),
    SectionType.OPERATING_ROOMS: Capacity(servers=50, queue=None),
    SectionType.WARD: Capacity(servers=40, queue=None),
    SectionType.ICU: Capacity(servers=10, queue=None),
    SectionType.CCU: Capacity(servers=5, queue=None),
    SectionType.RIP: Capacity(servers=1, queue=None)
}

REDUCTION_PERCENT= 0.2
electricity_suspension_hours = 24