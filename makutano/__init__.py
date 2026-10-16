"""Makutano: a machine learning pipeline for one Nairobi intersection.

see      video -> detections -> tracks -> per-approach arrivals and queues
predict  forecast each approach's arrivals from time of day and recent data
control  a reinforcement learning signal controller, trained and stress-tested
         in a SUMO model of the junction calibrated against the video
sim      SUMO demand generation, runs and calibration checks
"""
