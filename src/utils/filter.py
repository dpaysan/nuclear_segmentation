from src.selection.filtering import AreaFilter, AspectRatioFilter, SolidityFilter, EccentricityFilter, \
    ConfocalShiftFilter


def get_filter_from_config(filter_config):
    filter_type = filter_config.pop("type")
    if filter_type == 'area':
        return AreaFilter(properties=None, **filter_config)
    elif filter_type == 'aspect_ratio':
        return AspectRatioFilter(properties=None, **filter_config)
    elif filter_type == 'solidity':
        return SolidityFilter(properties=None, **filter_config)
    elif filter_type == 'eccentricity':
        return EccentricityFilter(properties=None, **filter_config)
    elif filter_type == 'confocal_shift':
        return ConfocalShiftFilter(**filter_config)
    elif filter_type == 'dead_cell':
        return DeadCellFilter(**filter_config)
    else:
        raise RuntimeError('Unknown filter: {}'.format(filter_type))
