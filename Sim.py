import time
from datetime import datetime

import numpy as np
import pyvista as pv

from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time


# ====================================================
# All dimensions in mm
# All mechanical angles in degrees
# ====================================================


# Site / Time =========================================

# Replace with the exact telescope location later
site_latitude_deg = 35.910057
site_longitude_deg = -79.052521      # East positive, West negative
site_height_m = 0

# None = current real time
#
# Or simulate a time:
# local_time = "2026-10-05T21:30:00"
#
# If no UTC offset is supplied, Python assumes the
# telescope computer's local timezone.
#
# You can also explicitly specify:
# local_time = "2026-10-05T21:30:00-04:00"

local_time = None


# Target ==============================================

# Celestial coordinates
# RA in HOURS
# Dec in DEGREES

target_ra_hours = 20.0
target_dec_deg = 40.0


# Fixed Mount / Tripod Dimensions =====================

# Height of idealized mount pivot above ground
tripod_height = 1000

# Central pier
pier_radius = 90

# Tripod legs
tripod_leg_radius = 30
tripod_leg_attach_height = 650
tripod_foot_radius = 650

# Rotation of tripod footprint around vertical axis
tripod_rotation_deg = 0

# Real-mount axis offsets
# Leave zero for idealized mount for now
dec_axes_offset = 0
ra_axes_offset = 0


# Dish Dimensions =====================================

dish_diameter = 2300
focal_length = 805

dish_radius = dish_diameter / 2
dish_depth = dish_radius**2 / (4 * focal_length)

# Distance from ideal mount pivot to dish vertex
# TEMP placeholder
dish_pivot_distance = 450


# Counterweight Dimensions ============================

# TEMP placeholders until measured

counterweight_shaft_length = 650
counterweight_shaft_radius = 18

counterweight_1_distance = 320
counterweight_2_distance = 500

counterweight_radius = 110
counterweight_thickness = 80


# Collision Settings ==================================

# Surface-to-surface clearance required
# before a move is considered safe

safety_margin = 75

# Number of intermediate configurations checked
# before a celestial slew is allowed

collision_path_samples = 150


# Mechanical State ====================================

# Ideal polar alignment
lat_offset = site_latitude_deg

# Your mechanical angle definitions

dec_angle = 0
# rotates the RA tangent plane around the NCP axis

ra_angle = 0
# rotates the telescope inside that plane

angle_step = 5


# Display / Status ====================================

scene_initialized = False
target_info = None

path_status = "No slew checked yet"
last_path_clearance = None


# Local Cartesian =====================================
#
# +x = East
# +y = North
# +z = Up

lat = np.deg2rad(lat_offset)


# North celestial pole
np_norm = np.array([
    0.0,
    np.cos(lat),
    np.sin(lat)
])


# Due east
east_norm = np.array([
    1.0,
    0.0,
    0.0
])


# Perpendicular to NCP and east
perp_norm = np.array([
    0.0,
    -np.sin(lat),
    np.cos(lat)
])


# Astronomy ===========================================

location = EarthLocation(
    lat=site_latitude_deg * u.deg,
    lon=site_longitude_deg * u.deg,
    height=site_height_m * u.m
)


def get_observation_time():

    # Real-time mode
    if local_time is None:
        return Time.now()

    # Simulated/manual time
    dt = datetime.fromisoformat(local_time)

    # If no UTC offset was included, assume the
    # computer's local timezone.
    if dt.tzinfo is None:
        dt = dt.astimezone()

    return Time(dt)


def celestial_to_direction(ra_hours, dec_deg):

    observation_time = get_observation_time()

    target = SkyCoord(
        ra=ra_hours * u.hourangle,
        dec=dec_deg * u.deg,
        frame="icrs"
    )

    altaz_frame = AltAz(
        obstime=observation_time,
        location=location
    )

    target_altaz = target.transform_to(
        altaz_frame
    )

    alt = target_altaz.alt.to_value(u.rad)
    az = target_altaz.az.to_value(u.rad)

    # Astropy azimuth:
    #
    # 0 deg  = North
    # 90 deg = East

    target_direction = np.array([
        np.cos(alt) * np.sin(az),   # East
        np.cos(alt) * np.cos(az),   # North
        np.sin(alt)                 # Up
    ])

    # Local apparent sidereal time
    lst = observation_time.sidereal_time(
        "apparent",
        longitude=location.lon
    )

    return (
        target_direction,
        target_altaz,
        observation_time,
        lst
    )


# Mechanical Kinematics ===============================

def get_mount_direction(
    ra_value=None,
    dec_value=None
):

    if ra_value is None:
        ra_value = ra_angle

    if dec_value is None:
        dec_value = dec_angle

    dec = np.deg2rad(dec_value)
    ra = np.deg2rad(ra_value)

    # Dec rotates the tangent plane through which
    # the telescope's RA motion occurs.
    ra_plane_direction = (
        np.cos(dec) * east_norm
        + np.sin(dec) * perp_norm
    )

    # RA rotates the telescope inside that plane.
    mount_direction = (
        np.cos(ra) * np_norm
        + np.sin(ra) * ra_plane_direction
    )

    # Axis normal to the RA motion plane.
    #
    # For our idealized model this is also used as
    # the counterweight-shaft direction.
    inner_axis = np.cross(
        np_norm,
        ra_plane_direction
    )

    inner_axis = (
        inner_axis
        / np.linalg.norm(inner_axis)
    )

    return (
        ra_plane_direction,
        mount_direction,
        inner_axis
    )


def angle_difference(a, b):
    """
    Smallest signed angular difference from b to a.
    """

    return (a - b + 180) % 360 - 180


def direction_to_mount_angles(target_direction):
    """
    Convert a local XYZ pointing vector into the two
    mechanical angles used by this simulator.

    There are two mechanically equivalent angle pairs
    for most pointing directions.

    Choose whichever one is closest to the current
    simulated mount configuration.
    """

    target = (
        target_direction
        / np.linalg.norm(target_direction)
    )

    pole_component = np.clip(
        np.dot(target, np_norm),
        -1.0,
        1.0
    )

    ra_magnitude = np.arccos(
        pole_component
    )

    east_component = np.dot(
        target,
        east_norm
    )

    perp_component = np.dot(
        target,
        perp_norm
    )

    # At NCP/SCP, the tangent-plane angle is undefined.
    if np.isclose(
        np.sin(ra_magnitude),
        0.0,
        atol=1e-8
    ):

        if pole_component > 0:
            return 0.0, dec_angle

        return 180.0, dec_angle

    plane_angle = np.arctan2(
        perp_component,
        east_component
    )

    ra1 = np.rad2deg(
        ra_magnitude
    )

    dec1 = np.rad2deg(
        plane_angle
    )

    # Equivalent solution
    ra2 = -ra1
    dec2 = dec1 + 180

    dec1 = (
        dec1 + 180
    ) % 360 - 180

    dec2 = (
        dec2 + 180
    ) % 360 - 180

    score1 = (
        angle_difference(
            ra1,
            ra_angle
        )**2
        +
        angle_difference(
            dec1,
            dec_angle
        )**2
    )

    score2 = (
        angle_difference(
            ra2,
            ra_angle
        )**2
        +
        angle_difference(
            dec2,
            dec_angle
        )**2
    )

    if score1 <= score2:
        return ra1, dec1

    return ra2, dec2


# Geometry Calculation ================================

mount_pivot = np.array([
    0.0,
    0.0,
    tripod_height
])


def get_mount_geometry(
    ra_value=None,
    dec_value=None
):

    (
        ra_plane_direction,
        mount_direction,
        counterweight_axis
    ) = get_mount_direction(
        ra_value,
        dec_value
    )

    # Dish vertex
    dish_vertex = (
        mount_pivot
        + dish_pivot_distance
        * mount_direction
    )

    # Feed / receiver position
    receiver_position = (
        dish_vertex
        + focal_length
        * mount_direction
    )

    # Counterweight shaft
    shaft_start = mount_pivot

    shaft_end = (
        mount_pivot
        + counterweight_shaft_length
        * counterweight_axis
    )

    # Counterweight centers
    counterweight_1_center = (
        mount_pivot
        + counterweight_1_distance
        * counterweight_axis
    )

    counterweight_2_center = (
        mount_pivot
        + counterweight_2_distance
        * counterweight_axis
    )

    return {
        "ra_plane_direction":
            ra_plane_direction,

        "mount_direction":
            mount_direction,

        "counterweight_axis":
            counterweight_axis,

        "dish_vertex":
            dish_vertex,

        "receiver_position":
            receiver_position,

        "shaft_start":
            shaft_start,

        "shaft_end":
            shaft_end,

        "counterweight_1_center":
            counterweight_1_center,

        "counterweight_2_center":
            counterweight_2_center
    }


# Tripod Geometry =====================================

def get_tripod_leg_segments():

    legs = []

    rotation = np.deg2rad(
        tripod_rotation_deg
    )

    for i in range(3):

        angle = (
            rotation
            + i * 2 * np.pi / 3
        )

        direction = np.array([
            np.cos(angle),
            np.sin(angle),
            0.0
        ])

        top = np.array([
            pier_radius
            * direction[0],

            pier_radius
            * direction[1],

            tripod_leg_attach_height
        ])

        bottom = np.array([
            tripod_foot_radius
            * direction[0],

            tripod_foot_radius
            * direction[1],

            0.0
        ])

        legs.append(
            (top, bottom)
        )

    return legs


tripod_leg_segments = (
    get_tripod_leg_segments()
)


# Distance / Collision Math ===========================

def point_segment_distance(
    point,
    segment_start,
    segment_end
):

    segment = (
        segment_end
        - segment_start
    )

    length_squared = np.dot(
        segment,
        segment
    )

    if np.isclose(
        length_squared,
        0.0
    ):
        return np.linalg.norm(
            point
            - segment_start
        )

    t = np.dot(
        point - segment_start,
        segment
    ) / length_squared

    t = np.clip(
        t,
        0.0,
        1.0
    )

    closest = (
        segment_start
        + t * segment
    )

    return np.linalg.norm(
        point - closest
    )


def segment_segment_distance(
    p1,
    q1,
    p2,
    q2
):
    """
    Minimum distance between two 3D line segments.
    """

    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    denominator = (
        a * c
        - b * b
    )

    small = 1e-12

    if denominator < small:

        s = 0.0

        if c > small:
            t = np.clip(
                e / c,
                0.0,
                1.0
            )
        else:
            t = 0.0

    else:

        s = np.clip(
            (b * e - c * d)
            / denominator,
            0.0,
            1.0
        )

        t = (
            b * s + e
        ) / c if c > small else 0.0

        if t < 0.0:

            t = 0.0

            s = np.clip(
                -d / a,
                0.0,
                1.0
            )

        elif t > 1.0:

            t = 1.0

            s = np.clip(
                (b - d) / a,
                0.0,
                1.0
            )

    closest_1 = (
        p1 + s * u
    )

    closest_2 = (
        p2 + t * v
    )

    return np.linalg.norm(
        closest_1
        - closest_2
    )


def check_pose_clearance(
    ra_value=None,
    dec_value=None
):
    """
    Conservative collision model.

    Counterweights are treated as spheres with radius
    equal to their physical cylinder radius.

    Counterweight shaft and tripod legs are treated as
    cylinders.

    Returns the minimum surface-to-surface clearance.
    """

    geometry = get_mount_geometry(
        ra_value,
        dec_value
    )

    minimum_clearance = np.inf
    closest_pair = None

    weight_centers = [
        (
            "Counterweight 1",
            geometry[
                "counterweight_1_center"
            ]
        ),
        (
            "Counterweight 2",
            geometry[
                "counterweight_2_center"
            ]
        )
    ]

    for leg_number, (
        leg_start,
        leg_end
    ) in enumerate(
        tripod_leg_segments,
        start=1
    ):

        # --------------------------------------------
        # Counterweights vs tripod legs
        # --------------------------------------------

        for (
            weight_name,
            weight_center
        ) in weight_centers:

            center_distance = (
                point_segment_distance(
                    weight_center,
                    leg_start,
                    leg_end
                )
            )

            surface_clearance = (
                center_distance
                - counterweight_radius
                - tripod_leg_radius
            )

            if (
                surface_clearance
                < minimum_clearance
            ):

                minimum_clearance = (
                    surface_clearance
                )

                closest_pair = (
                    f"{weight_name} / "
                    f"Tripod leg {leg_number}"
                )

        # --------------------------------------------
        # Counterweight shaft vs tripod legs
        # --------------------------------------------

        shaft_distance = (
            segment_segment_distance(
                geometry["shaft_start"],
                geometry["shaft_end"],
                leg_start,
                leg_end
            )
        )

        shaft_clearance = (
            shaft_distance
            - counterweight_shaft_radius
            - tripod_leg_radius
        )

        if (
            shaft_clearance
            < minimum_clearance
        ):

            minimum_clearance = (
                shaft_clearance
            )

            closest_pair = (
                f"Counterweight shaft / "
                f"Tripod leg {leg_number}"
            )

    unsafe = (
        minimum_clearance
        < safety_margin
    )

    return {
        "unsafe": unsafe,
        "clearance": minimum_clearance,
        "closest_pair": closest_pair
    }


def check_slew_path(
    target_ra_angle,
    target_dec_angle
):

    start_ra = ra_angle
    start_dec = dec_angle

    delta_ra = angle_difference(
        target_ra_angle,
        start_ra
    )

    delta_dec = angle_difference(
        target_dec_angle,
        start_dec
    )

    minimum_clearance = np.inf
    closest_pair = None
    closest_fraction = 0.0

    for fraction in np.linspace(
        0.0,
        1.0,
        collision_path_samples
    ):

        test_ra = (
            start_ra
            + delta_ra
            * fraction
        )

        test_dec = (
            start_dec
            + delta_dec
            * fraction
        )

        result = check_pose_clearance(
            test_ra,
            test_dec
        )

        if (
            result["clearance"]
            < minimum_clearance
        ):

            minimum_clearance = (
                result["clearance"]
            )

            closest_pair = (
                result["closest_pair"]
            )

            closest_fraction = (
                fraction
            )

        if result["unsafe"]:

            return {
                "safe": False,
                "clearance":
                    minimum_clearance,

                "closest_pair":
                    closest_pair,

                "fraction":
                    closest_fraction,

                "ra":
                    test_ra,

                "dec":
                    test_dec
            }

    return {
        "safe": True,
        "clearance":
            minimum_clearance,

        "closest_pair":
            closest_pair,

        "fraction":
            closest_fraction
    }


# Rotation Math =======================================

def rotation_from_z(direction):

    direction = (
        direction
        / np.linalg.norm(direction)
    )

    z_axis = np.array([
        0.0,
        0.0,
        1.0
    ])

    cross = np.cross(
        z_axis,
        direction
    )

    dot = np.dot(
        z_axis,
        direction
    )

    if np.isclose(
        dot,
        1.0
    ):
        return np.eye(3)

    if np.isclose(
        dot,
        -1.0
    ):

        return np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

    cross /= np.linalg.norm(
        cross
    )

    x, y, z = cross

    K = np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

    angle = np.arccos(
        dot
    )

    return (
        np.eye(3)
        + np.sin(angle) * K
        + (
            1 - np.cos(angle)
        ) * (K @ K)
    )


# Mesh Helpers ========================================

def make_cylinder_between(
    start,
    end,
    radius
):

    start = np.asarray(
        start,
        dtype=float
    )

    end = np.asarray(
        end,
        dtype=float
    )

    vector = end - start

    length = np.linalg.norm(
        vector
    )

    if np.isclose(
        length,
        0.0
    ):
        return None

    direction = (
        vector / length
    )

    center = (
        start + end
    ) / 2

    return pv.Cylinder(
        center=center,
        direction=direction,
        radius=radius,
        height=length
    )


# Dish Geometry =======================================

def make_dish():

    radial_points = 30
    angular_points = 80

    r = np.linspace(
        0,
        dish_radius,
        radial_points
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        angular_points
    )

    R, Theta = np.meshgrid(
        r,
        theta,
        indexing="ij"
    )

    X = (
        R * np.cos(Theta)
    )

    Y = (
        R * np.sin(Theta)
    )

    Z = (
        R**2
        / (4 * focal_length)
    )

    return pv.StructuredGrid(
        X,
        Y,
        Z
    )


dish_original = make_dish()

dish_local_points = (
    dish_original.points.copy()
)


receiver_original = pv.Sphere(
    radius=50,
    center=(
        0,
        0,
        focal_length
    )
)

receiver_local_points = (
    receiver_original.points.copy()
)


# Plot =================================================

plotter = pv.Plotter()

plotter.set_background(
    "white"
)


def draw():

    global scene_initialized

    if scene_initialized:
        camera_position = (
            plotter.camera_position
        )
    else:
        camera_position = None

    plotter.clear()

    geometry = get_mount_geometry()

    mount_direction = (
        geometry[
            "mount_direction"
        ]
    )

    ra_plane_direction = (
        geometry[
            "ra_plane_direction"
        ]
    )

    counterweight_axis = (
        geometry[
            "counterweight_axis"
        ]
    )

    dish_vertex = (
        geometry[
            "dish_vertex"
        ]
    )

    # --------------------------------------------
    # Current safety state
    # --------------------------------------------

    safety = (
        check_pose_clearance()
    )

    # --------------------------------------------
    # Ground
    # --------------------------------------------

    ground = pv.Disc(
        center=(0, 0, 0),
        inner=0,
        outer=tripod_foot_radius
        * 1.35,
        normal=(0, 0, 1)
    )

    plotter.add_mesh(
        ground,
        color="gainsboro",
        opacity=0.25
    )

    # --------------------------------------------
    # Pier
    # --------------------------------------------

    pier = pv.Cylinder(
        center=(
            0,
            0,
            tripod_height / 2
        ),
        direction=(
            0,
            0,
            1
        ),
        radius=pier_radius,
        height=tripod_height
    )

    plotter.add_mesh(
        pier,
        color="dimgray"
    )

    # --------------------------------------------
    # Tripod legs
    # --------------------------------------------

    for (
        leg_start,
        leg_end
    ) in tripod_leg_segments:

        leg_mesh = (
            make_cylinder_between(
                leg_start,
                leg_end,
                tripod_leg_radius
            )
        )

        plotter.add_mesh(
            leg_mesh,
            color="gray"
        )

    # --------------------------------------------
    # Mount pivot / body
    # --------------------------------------------

    plotter.add_mesh(
        pv.Sphere(
            radius=130,
            center=mount_pivot
        ),
        color="slategray"
    )

    # --------------------------------------------
    # Dish support / boom
    # --------------------------------------------

    dish_support = (
        make_cylinder_between(
            mount_pivot,
            dish_vertex,
            35
        )
    )

    if dish_support is not None:

        plotter.add_mesh(
            dish_support,
            color="darkgray"
        )

    # --------------------------------------------
    # Dish
    # --------------------------------------------

    rotation = rotation_from_z(
        mount_direction
    )

    dish = dish_original.copy()

    dish.points = (
        dish_local_points
        @ rotation.T
        + dish_vertex
    )

    plotter.add_mesh(
        dish,
        color="lightgray",
        show_edges=True,
        opacity=0.9
    )

    # --------------------------------------------
    # Receiver / feed
    # --------------------------------------------

    receiver = (
        receiver_original.copy()
    )

    receiver.points = (
        receiver_local_points
        @ rotation.T
        + dish_vertex
    )

    plotter.add_mesh(
        receiver,
        color="orange"
    )

    # --------------------------------------------
    # Counterweight shaft
    # --------------------------------------------

    shaft_mesh = (
        make_cylinder_between(
            geometry[
                "shaft_start"
            ],
            geometry[
                "shaft_end"
            ],
            counterweight_shaft_radius
        )
    )

    if shaft_mesh is not None:

        plotter.add_mesh(
            shaft_mesh,
            color="black"
        )

    # --------------------------------------------
    # Counterweights
    # --------------------------------------------

    for center in [
        geometry[
            "counterweight_1_center"
        ],
        geometry[
            "counterweight_2_center"
        ]
    ]:

        weight = pv.Cylinder(
            center=center,
            direction=
                counterweight_axis,
            radius=
                counterweight_radius,
            height=
                counterweight_thickness
        )

        plotter.add_mesh(
            weight,
            color="darkslategray"
        )

    # --------------------------------------------
    # Direction arrows
    # --------------------------------------------

    arrow_length = (
        dish_diameter
        * 0.65
    )

    origin = np.array([
        mount_pivot
    ])

    # NCP
    plotter.add_arrows(
        origin,
        np.array([
            np_norm
            * arrow_length
        ]),
        color="red"
    )

    # RA plane tangent direction
    plotter.add_arrows(
        origin,
        np.array([
            ra_plane_direction
            * arrow_length
        ]),
        color="green"
    )

    # Telescope pointing direction
    plotter.add_arrows(
        origin,
        np.array([
            mount_direction
            * arrow_length
        ]),
        color="blue"
    )

    # Counterweight / inner rotation axis
    plotter.add_arrows(
        origin,
        np.array([
            counterweight_axis
            * arrow_length
            * 0.6
        ]),
        color="black"
    )

    # --------------------------------------------
    # Requested celestial target
    # --------------------------------------------

    if target_info is not None:

        target_direction = (
            target_info[
                "direction"
            ]
        )

        plotter.add_arrows(
            origin,
            np.array([
                target_direction
                * arrow_length
            ]),
            color="purple"
        )

    # --------------------------------------------
    # World-coordinate axes
    # --------------------------------------------

    axis_length = (
        dish_diameter
    )

    plotter.add_lines(
        np.array([
            [0, 0, 0],
            [axis_length, 0, 0]
        ]),
        color="gray"
    )

    plotter.add_lines(
        np.array([
            [0, 0, 0],
            [0, axis_length, 0]
        ]),
        color="gray"
    )

    plotter.add_lines(
        np.array([
            [0, 0, 0],
            [0, 0, axis_length]
        ]),
        color="gray"
    )

    plotter.add_point_labels(
        np.array([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ]),
        [
            "East",
            "North",
            "Up"
        ],
        point_size=0,
        font_size=14
    )

    # --------------------------------------------
    # Status text
    # --------------------------------------------

    if safety["unsafe"]:

        safety_text = (
            "UNSAFE\n"
            f"Clearance: "
            f"{safety['clearance']:.1f} mm\n"
            f"{safety['closest_pair']}"
        )

    else:

        safety_text = (
            "SAFE\n"
            f"Clearance: "
            f"{safety['clearance']:.1f} mm\n"
            f"{safety['closest_pair']}"
        )

    status = (
        f"Mechanical RA: "
        f"{ra_angle:.1f} deg\n"

        f"Mechanical Dec: "
        f"{dec_angle:.1f} deg\n"

        f"\n"

        f"Target RA: "
        f"{target_ra_hours:.3f} h\n"

        f"Target Dec: "
        f"{target_dec_deg:.3f} deg\n"

        f"\n"

        f"{safety_text}\n"

        f"\n"

        f"Last path check:\n"
        f"{path_status}"
    )

    if target_info is not None:

        status += (
            f"\n\n"
            f"Altitude: "
            f"{target_info['alt']:.2f} deg\n"

            f"Azimuth: "
            f"{target_info['az']:.2f} deg\n"

            f"LST: "
            f"{target_info['lst']:.3f} h"
        )

    plotter.add_text(
        status,
        position="upper_left",
        font_size=10,
        color="black"
    )

    # --------------------------------------------
    # Camera
    # --------------------------------------------

    if camera_position is not None:

        plotter.camera_position = (
            camera_position
        )

    else:

        plotter.camera_position = (
            "iso"
        )

    plotter.render()

    scene_initialized = True


# Slew ================================================

def slew_to_angles(
    target_ra_angle,
    target_dec_angle,
    steps=60
):

    global ra_angle
    global dec_angle

    start_ra = ra_angle
    start_dec = dec_angle

    delta_ra = angle_difference(
        target_ra_angle,
        start_ra
    )

    delta_dec = angle_difference(
        target_dec_angle,
        start_dec
    )

    for i in range(
        1,
        steps + 1
    ):

        fraction = (
            i / steps
        )

        ra_angle = (
            start_ra
            + delta_ra
            * fraction
        )

        dec_angle = (
            start_dec
            + delta_dec
            * fraction
        )

        draw()

        plotter.update()

        time.sleep(
            0.02
        )


def goto_target():

    global target_info
    global path_status
    global last_path_clearance

    (
        direction,
        altaz,
        observation_time,
        lst
    ) = celestial_to_direction(
        target_ra_hours,
        target_dec_deg
    )

    (
        target_ra_angle,
        target_dec_angle
    ) = direction_to_mount_angles(
        direction
    )

    target_info = {

        "direction":
            direction,

        "alt":
            altaz.alt.to_value(
                u.deg
            ),

        "az":
            altaz.az.to_value(
                u.deg
            ),

        "lst":
            lst.to_value(
                u.hourangle
            )
    }

    print()
    print(
        "Celestial target"
    )

    print(
        f"RA:  "
        f"{target_ra_hours:.4f} h"
    )

    print(
        f"Dec: "
        f"{target_dec_deg:.4f} deg"
    )

    print()

    print(
        "Local sidereal time:",
        f"{target_info['lst']:.4f} h"
    )

    print(
        "Altitude:",
        f"{target_info['alt']:.2f} deg"
    )

    print(
        "Azimuth:",
        f"{target_info['az']:.2f} deg"
    )

    print()

    print(
        "Mechanical target:",
        f"RA angle = "
        f"{target_ra_angle:.2f} deg,",

        f"Dec angle = "
        f"{target_dec_angle:.2f} deg"
    )

    if target_info["alt"] < 0:

        print(
            "WARNING: target is "
            "below the horizon."
        )

    # --------------------------------------------
    # Check whole proposed path first
    # --------------------------------------------

    path_check = check_slew_path(
        target_ra_angle,
        target_dec_angle
    )

    last_path_clearance = (
        path_check[
            "clearance"
        ]
    )

    if not path_check["safe"]:

        path_status = (
            "REJECTED\n"
            f"Min clearance: "
            f"{path_check['clearance']:.1f} mm\n"
            f"{path_check['closest_pair']}\n"
            f"At "
            f"{100 * path_check['fraction']:.1f}% "
            f"of slew"
        )

        print()

        print(
            "SLEW REJECTED"
        )

        print(
            "Minimum clearance:",
            f"{path_check['clearance']:.1f} mm"
        )

        print(
            "Closest objects:",
            path_check[
                "closest_pair"
            ]
        )

        print(
            "Path position:",
            f"{100 * path_check['fraction']:.1f}%"
        )

        draw()

        return

    path_status = (
        "SAFE\n"
        f"Min clearance: "
        f"{path_check['clearance']:.1f} mm"
    )

    print()

    print(
        "Slew path safe."
    )

    print(
        "Minimum clearance:",
        f"{path_check['clearance']:.1f} mm"
    )

    slew_to_angles(
        target_ra_angle,
        target_dec_angle
    )


# Keyboard Controls ===================================

def ra_up():

    global ra_angle

    ra_angle += (
        angle_step
    )

    draw()


def ra_down():

    global ra_angle

    ra_angle -= (
        angle_step
    )

    draw()


def dec_right():

    global dec_angle

    dec_angle += (
        angle_step
    )

    draw()


def dec_left():

    global dec_angle

    dec_angle -= (
        angle_step
    )

    draw()


# Remove PyVista's default arrow-key camera behavior

plotter.iren.clear_events_for_key(
    "Up"
)

plotter.iren.clear_events_for_key(
    "Down"
)

plotter.iren.clear_events_for_key(
    "Left"
)

plotter.iren.clear_events_for_key(
    "Right"
)


plotter.add_key_event(
    "Up",
    ra_up
)

plotter.add_key_event(
    "Down",
    ra_down
)

plotter.add_key_event(
    "Right",
    dec_right
)

plotter.add_key_event(
    "Left",
    dec_left
)

# Press G to calculate current celestial target,
# check the entire slew path, and move only if safe.

plotter.add_key_event(
    "g",
    goto_target
)


# Start ===============================================

draw()

plotter.show()