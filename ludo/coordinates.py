"""
Board coordinate mappings for Ludo pieces.

Coordinates are in pygame coordinate system (top-left at 0,0).
x increases to the right, y increases downward.

Main track: 52 shared positions (0-51)
Home tracks: 6 positions per player (52-57, indexed as 0-5 in code)
Yard: 4 starting positions per player
"""

# Main track positions (0-51) - shared by all players
MAIN_TRACK_POSITIONS = {
    0: (783, 1624),  # Safe square
    1: (783, 1502),
    2: (783, 1381),
    3: (783, 1260),
    4: (783, 1143),
    5: (661, 1024),
    6: (545, 1024),
    7: (424, 1024),
    8: (305, 1024),  # Safe square
    9: (183, 1024),
    10: (60, 1023),
    11: (60, 901),
    12: (62, 785),
    13: (184, 785),  # Safe square
    14: (304, 785),
    15: (423, 785),
    16: (544, 787),
    17: (663, 785),
    18: (782, 663),
    19: (782, 542),
    20: (782, 421),
    21: (782, 302),  # Safe square
    22: (782, 182),
    23: (786, 61),
    24: (908, 61),
    25: (1025, 62),
    26: (1025, 184),  # Safe square
    27: (1025, 306),
    28: (1025, 426),
    29: (1025, 545),
    30: (1023, 665),
    31: (1147, 781),
    32: (1264, 781),
    33: (1389, 781),
    34: (1501, 781),  # Safe square
    35: (1626, 780),
    36: (1741, 781),
    37: (1741, 903),
    38: (1741, 1023),
    39: (1625, 1023),  # Safe square
    40: (1500, 1023),
    41: (1383, 1023),
    42: (1266, 1023),
    43: (1146, 1023),
    44: (1023, 1145),
    45: (1023, 1267),
    46: (1023, 1382),
    47: (1023, 1506),  # Safe square
    48: (1023, 1629),
    49: (1022, 1745),
    50: (903, 1745),
    51: (783, 1745),
}

# Home track positions (52-57, indexed as 0-5 in code) for each player
HOME_TRACK_POSITIONS = {
    "player_0": {  # Green
        0: (902, 1625),  # Position 52
        1: (902, 1504),  # Position 53
        2: (902, 1383),  # Position 54
        3: (902, 1262),  # Position 55
        4: (902, 1143),  # Position 56
        5: (902, 1014),  # Position 57 (finished)
    },
    "player_1": {  # Yellow
        0: (186, 903),  # Position 52
        1: (304, 903),  # Position 53
        2: (423, 903),  # Position 54
        3: (544, 903),  # Position 55
        4: (663, 903),  # Position 56
        5: (776, 903),  # Position 57 (finished)
    },
    "player_2": {  # Blue
        0: (902, 182),  # Position 52
        1: (902, 305),  # Position 53
        2: (902, 421),  # Position 54
        3: (902, 540),  # Position 55
        4: (902, 662),  # Position 56
        5: (902, 786),  # Position 57 (finished)
    },
    "player_3": {  # Red
        0: (1624, 902),  # Position 52
        1: (1502, 902),  # Position 53
        2: (1383, 902),  # Position 54
        3: (1268, 900),  # Position 55
        4: (1146, 904),  # Position 56
        5: (1024, 904),  # Position 57 (finished)
    },
}

# Yard (starting area) positions - 4 pieces per player
YARD_POSITIONS = {
    "player_0": [  # Green
        (241, 1315),   # Piece 0
        (468, 1315),   # Piece 1
        (468, 1537),   # Piece 2
        (244, 1537),   # Piece 3
    ],
    "player_1": [  # Yellow
        (244, 462),   # Piece 0
        (468, 462),   # Piece 1
        (468, 238),   # Piece 2
        (241, 238),   # Piece 3
    ],
    "player_2": [  # Blue
        (1331, 242),   # Piece 0
        (1555, 242),   # Piece 1
        (1555, 466),   # Piece 2
        (1333, 466),   # Piece 3
    ],
    "player_3": [  # Red
        (1324, 1330),   # Piece 0
        (1546, 1330),   # Piece 1
        (1546, 1546),   # Piece 2
        (1324, 1546),   # Piece 3
    ],
}
