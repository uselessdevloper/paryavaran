from enum import Enum

class LAND_TYPE(Enum):
    """Land type categories for Bhubaneswar, India.

    Replaces the UK Ordnance Survey Zoomstack categories.
    Sources: ESA WorldCover 2021, OpenStreetMap, Google Open Buildings.
    """

    GREENSPACE       = "Green_Space"
    WATER            = "Water"
    BUILDING         = "Building"
    ROAD             = "Road"
    RAILWAY          = "Railway"
    AIRPORT          = "Airport"
    HOSPITAL         = "Hospital"
    SCHOOL           = "School"
    BARE_LAND        = "Bare_Land"
    CROPLAND         = "Cropland"
    WETLAND          = "Wetland"
    # GOVERNMENT_LAND  = "Government_Land"  # enable when authoritative parcel data is available