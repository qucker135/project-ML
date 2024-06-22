# Dropped because they are identifiers, duplicate data or single value columns:
dropped_columns = [
    'Neo Reference ID',
    'Name',
    'Est Dia in KM(min)',
    'Est Dia in KM(max)',
    'Est Dia in Miles(min)',
    'Est Dia in Miles(max)',
    'Est Dia in Feet(min)',
    'Est Dia in Feet(max)',
    'Close Approach Date',
    'Relative Velocity km per hr',
    'Miles per hour',
    'Miss Dist.(miles)',
    'Miss Dist.(kilometers)',
    'Miss Dist.(Astronomical)',
    'Orbiting Body',
    'Orbit ID',
    'Equinox',
]

# Potentially unnecessary columns:
unnecessary_columns = [
    'Absolute Magnitude',
    'Orbit Determination Date',
]