"""The project's road-user classes, used by every module.

These are the six classes the proposal names for Kenyan mixed traffic. The
detector is fine-tuned to output exactly these names; measurement, SUMO vehicle
types and the live console use the same names, so counts and simulated flows
line up class by class.
"""

MATATU = "matatu"
BODA_BODA = "boda_boda"
PRIVATE_CAR = "private_car"
TRUCK_BUS = "truck_bus"      # trucks and large buses share one class
HANDCART = "handcart"
PEDESTRIAN = "pedestrian"

VEHICLE_CLASSES = (MATATU, BODA_BODA, PRIVATE_CAR, TRUCK_BUS, HANDCART)
CLASSES = VEHICLE_CLASSES + (PEDESTRIAN,)

# Pre-trained COCO weights know none of matatu or handcart. Their buses and trucks
# both become truck_bus, so matatus detected as buses land there too; telling them
# apart is what fine-tuning on local footage is for. COCO bicycles are not one of
# the six classes and are ignored.
COCO_TO_LOCAL = {
    0: PEDESTRIAN,
    2: PRIVATE_CAR,
    3: BODA_BODA,
    5: TRUCK_BUS,
    7: TRUCK_BUS,
}
