"""Label set of the fine-tuning task: route a driver's problem description to the right car system.

Business need: a garage (or a repair forum) receives free-text messages like "squeaking noise when I stop"
and someone sorts them by hand to the right mechanic / section. The descriptions below are used by the
zero-shot baseline and by the LLM prompt baseline, and they document the labelling guidelines.
"""
from __future__ import annotations

LABELS: dict[str, str] = {
    "engine": "Engine and fuel system: engine noises, misfires, oil leaks, check engine light, timing belt, "
              "spark plugs, fuel pump, exhaust, poor acceleration, stalling.",
    "brakes": "Brakes: brake pads, discs/rotors, calipers, brake fluid, squealing or grinding when braking, "
              "soft brake pedal, ABS warning, parking brake.",
    "electrical": "Electrical system: battery, alternator, starter, fuses, lights, headlights, power windows, "
                  "central locking, dashboard warning electronics, car won't start with clicking.",
    "cooling_hvac": "Cooling and climate: overheating, coolant leaks, radiator, thermostat, water pump, "
                    "air conditioning, heater, blower fan, cabin air filter.",
    "suspension_steering": "Suspension, steering, wheels and tires: shocks, struts, control arms, ball joints, "
                           "tie rods, power steering, alignment, vibrations at speed, clunks over bumps, tires.",
    "body_interior": "Body and interior: doors, door handles, mirrors, windshield wipers, seats, trim, "
                     "locks mechanics, hood, trunk, interior panels, glove box.",
}

LABEL_NAMES: list[str] = list(LABELS)
