from src.selection.filtering import (
    ObjectAreaFilter,
    ObjectAspectRatioFilter,
    ObjectSolidityFilter,
    ConfocalShiftFilter,
    ConservativeDeadCellFilter,
    ObjectEccentricityFilter,
)


def get_filter_from_config(filter_config):
    filter_type = filter_config.pop("type")
    if filter_type == "area":
        return ObjectAreaFilter(properties=None, **filter_config)
    elif filter_type == "aspect_ratio":
        return ObjectAspectRatioFilter(properties=None, **filter_config)
    elif filter_type == "solidity":
        return ObjectSolidityFilter(properties=None, **filter_config)
    elif filter_type == "eccentricity":
        return ObjectEccentricityFilter(properties=None, **filter_config)
    elif filter_type == "confocal_shift":
        return ConfocalShiftFilter(**filter_config)
    elif filter_type == "dead_cell":
        return ConservativeDeadCellFilter(**filter_config)
    else:
        raise RuntimeError("Unknown filter: {}".format(filter_type))
