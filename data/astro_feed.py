"""
AstroFeed - Vedic astrology data feed for market analysis.
Uses Kerykeion for planetary calculations, Mumbai locale.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

try:
    from kerykeion import AstrologicalSubject
except ImportError:
    logger.warning("kerykeion not installed. Run: pip install kerykeion")
    AstrologicalSubject = None


NAKSHATRA_LIST = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira",
    "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha",
    "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati",
    "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha",
    "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
    "Uttara Bhadrapada", "Revati",
]

NAKSHATRA_NATURE = {
    "Ashwini": "bullish",
    "Bharani": "bearish",
    "Krittika": "volatile",
    "Rohini": "bullish",
    "Mrigashira": "neutral",
    "Ardra": "bearish",
    "Punarvasu": "bullish",
    "Pushya": "bullish",
    "Ashlesha": "bearish",
    "Magha": "volatile",
    "Purva Phalguni": "bullish",
    "Uttara Phalguni": "neutral",
    "Hasta": "bullish",
    "Chitra": "neutral",
    "Swati": "bullish",
    "Vishakha": "volatile",
    "Anuradha": "bullish",
    "Jyeshtha": "bearish",
    "Mula": "bearish",
    "Purva Ashadha": "volatile",
    "Uttara Ashadha": "neutral",
    "Shravana": "bullish",
    "Dhanishta": "bullish",
    "Shatabhisha": "volatile",
    "Purva Bhadrapada": "bearish",
    "Uttara Bhadrapada": "neutral",
    "Revati": "bullish",
}

NAKSHATRA_LORD = {
    "Ashwini": "Ketu",
    "Bharani": "Venus",
    "Krittika": "Sun",
    "Rohini": "Moon",
    "Mrigashira": "Mars",
    "Ardra": "Rahu",
    "Punarvasu": "Jupiter",
    "Pushya": "Saturn",
    "Ashlesha": "Mercury",
    "Magha": "Ketu",
    "Purva Phalguni": "Venus",
    "Uttara Phalguni": "Sun",
    "Hasta": "Moon",
    "Chitra": "Mars",
    "Swati": "Rahu",
    "Vishakha": "Jupiter",
    "Anuradha": "Saturn",
    "Jyeshtha": "Mercury",
    "Mula": "Ketu",
    "Purva Ashadha": "Venus",
    "Uttara Ashadha": "Sun",
    "Shravana": "Moon",
    "Dhanishta": "Mars",
    "Shatabhisha": "Rahu",
    "Purva Bhadrapada": "Jupiter",
    "Uttara Bhadrapada": "Saturn",
    "Revati": "Mercury",
}

TITHI_NAMES = [
    "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami",
    "Shashthi", "Saptami", "Ashtami", "Navami", "Dashami",
    "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi", "Purnima",
    "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami",
    "Shashthi", "Saptami", "Ashtami", "Navami", "Dashami",
    "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi", "Amavasya",
]

TITHI_NATURE = {
    1: "bullish",       # Shukla Pratipada - new beginnings
    2: "bullish",       # Shukla Dwitiya
    3: "bullish",       # Shukla Tritiya
    4: "bearish",       # Chaturthi - Vinayaka
    5: "bullish",       # Panchami - Lakshmi
    6: "neutral",       # Shashthi
    7: "bullish",       # Saptami
    8: "bearish",       # Ashtami - volatile
    9: "bearish",       # Navami - volatile
    10: "bullish",      # Dashami - victory
    11: "bullish",      # Ekadashi - auspicious
    12: "neutral",      # Dwadashi
    13: "neutral",      # Trayodashi
    14: "bearish",      # Chaturdashi - Shiva
    15: "volatile",     # Purnima - full moon
    16: "bearish",      # Krishna Pratipada - decline begins
    17: "bearish",      # Krishna Dwitiya
    18: "neutral",      # Krishna Tritiya
    19: "bearish",      # Krishna Chaturthi
    20: "neutral",      # Krishna Panchami
    21: "neutral",      # Krishna Shashthi
    22: "bearish",      # Krishna Saptami
    23: "bearish",      # Krishna Ashtami - very volatile
    24: "bearish",      # Krishna Navami
    25: "neutral",      # Krishna Dashami
    26: "bullish",      # Krishna Ekadashi - auspicious
    27: "neutral",      # Krishna Dwadashi
    28: "bearish",      # Krishna Trayodashi
    29: "bearish",      # Krishna Chaturdashi
    30: "volatile",     # Amavasya - new moon
}

DAY_LORDS = {
    0: "Moon",      # Monday
    1: "Mars",      # Tuesday
    2: "Mercury",   # Wednesday
    3: "Jupiter",   # Thursday
    4: "Venus",     # Friday
    5: "Saturn",    # Saturday
    6: "Sun",       # Sunday
}

# Chaldean hora sequence
HORA_SEQUENCE = ["Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon"]

BENEFIC_PLANETS = {"Jupiter", "Venus", "Mercury", "Moon"}
MALEFIC_PLANETS = {"Saturn", "Mars", "Sun", "Rahu", "Ketu"}

SIGN_LIST = [
    "Ari", "Tau", "Gem", "Can", "Leo", "Vir",
    "Lib", "Sco", "Sag", "Cap", "Aqu", "Pis",
]

SIGN_FULL_NAMES = {
    "Ari": "Aries", "Tau": "Taurus", "Gem": "Gemini", "Can": "Cancer",
    "Leo": "Leo", "Vir": "Virgo", "Lib": "Libra", "Sco": "Scorpio",
    "Sag": "Sagittarius", "Cap": "Capricorn", "Aqu": "Aquarius", "Pis": "Pisces",
}

ASPECT_TYPES = {
    "Conjunction": {"angle": 0, "orb": 8},
    "Sextile": {"angle": 60, "orb": 6},
    "Square": {"angle": 90, "orb": 6},
    "Trine": {"angle": 120, "orb": 6},
    "Opposition": {"angle": 180, "orb": 8},
}

ASPECT_MARKET_IMPACT = {
    "Conjunction": "strong_trend",
    "Sextile": "mildly_bullish",
    "Square": "volatile",
    "Trine": "bullish",
    "Opposition": "reversal",
}

# Planet attribute names in kerykeion
KERYKEION_PLANET_MAP = {
    "sun": "sun",
    "moon": "moon",
    "mercury": "mercury",
    "venus": "venus",
    "mars": "mars",
    "jupiter": "jupiter",
    "saturn": "saturn",
}


class AstroFeed:
    """Vedic astrology data feed using Kerykeion for planetary calculations."""

    MUMBAI_LAT = 19.0760
    MUMBAI_LNG = 72.8777
    MUMBAI_TZ = "Asia/Kolkata"

    def __init__(self):
        logger.info("AstroFeed initialized | locale=Mumbai, IN")

    # ------------------------------------------------------------------
    # 1. Planet positions
    # ------------------------------------------------------------------
    def get_planet_positions(self, dt: datetime) -> Dict:
        """
        Compute positions of 9 Vedic planets for a given datetime.
        Uses Kerykeion AstrologicalSubject centred on Mumbai.

        Returns dict keyed by planet name, each containing:
          degree (0-360 absolute), sign, retrograde (bool)
        """
        try:
            subject = AstrologicalSubject(
                "Market",
                dt.year, dt.month, dt.day,
                dt.hour, dt.minute,
                lng=self.MUMBAI_LNG,
                lat=self.MUMBAI_LAT,
                tz_str=self.MUMBAI_TZ,
                city="Mumbai",
                nation="IN",
            )

            positions: Dict = {}

            for planet_name, attr_name in KERYKEION_PLANET_MAP.items():
                planet_obj = getattr(subject, attr_name, None)
                if planet_obj is None:
                    logger.warning(f"Planet attribute '{attr_name}' not found on subject")
                    continue

                # Kerykeion stores absolute longitude in .abs_pos or .position
                abs_deg = getattr(planet_obj, "abs_pos", None)
                if abs_deg is None:
                    abs_deg = getattr(planet_obj, "position", 0.0)

                sign_short = getattr(planet_obj, "sign", "")
                sign_full = SIGN_FULL_NAMES.get(sign_short, sign_short)
                retro = getattr(planet_obj, "retrograde", False)

                positions[planet_name] = {
                    "degree": round(float(abs_deg), 4),
                    "sign": sign_full,
                    "retrograde": bool(retro),
                }

            # Rahu & Ketu (lunar nodes) - Kerykeion uses mean_node
            # Rahu = True Node / Mean Node; Ketu = Rahu + 180
            node_obj = getattr(subject, "mean_node", None)
            if node_obj is None:
                node_obj = getattr(subject, "true_node", None)

            if node_obj is not None:
                rahu_deg = getattr(node_obj, "abs_pos", None)
                if rahu_deg is None:
                    rahu_deg = getattr(node_obj, "position", 0.0)
                rahu_deg = float(rahu_deg)
                ketu_deg = (rahu_deg + 180.0) % 360.0

                rahu_sign_idx = int(rahu_deg // 30)
                ketu_sign_idx = int(ketu_deg // 30)

                positions["rahu"] = {
                    "degree": round(rahu_deg, 4),
                    "sign": SIGN_FULL_NAMES.get(SIGN_LIST[rahu_sign_idx], SIGN_LIST[rahu_sign_idx]),
                    "retrograde": True,  # Rahu is always retrograde
                }
                positions["ketu"] = {
                    "degree": round(ketu_deg, 4),
                    "sign": SIGN_FULL_NAMES.get(SIGN_LIST[ketu_sign_idx], SIGN_LIST[ketu_sign_idx]),
                    "retrograde": True,  # Ketu is always retrograde
                }
            else:
                logger.warning("Lunar nodes not found in kerykeion subject")

            logger.info(f"Planet positions computed | dt={dt.isoformat()} | planets={len(positions)}")
            return positions

        except Exception as e:
            logger.error(f"get_planet_positions failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # 2. Nakshatra
    # ------------------------------------------------------------------
    def get_nakshatra(self, degree: float) -> Dict:
        """
        Determine nakshatra from absolute Moon degree (0-360).
        27 nakshatras, each spanning 13.3333 degrees, 4 padas each.
        """
        try:
            degree = float(degree) % 360.0
            nakshatra_span = 360.0 / 27.0  # 13.3333...
            pada_span = nakshatra_span / 4.0  # 3.3333...

            nakshatra_idx = int(degree / nakshatra_span)
            nakshatra_idx = min(nakshatra_idx, 26)

            offset_in_nakshatra = degree - (nakshatra_idx * nakshatra_span)
            pada = int(offset_in_nakshatra / pada_span) + 1
            pada = min(pada, 4)

            name = NAKSHATRA_LIST[nakshatra_idx]
            nature = NAKSHATRA_NATURE.get(name, "neutral")
            lord = NAKSHATRA_LORD.get(name, "Unknown")

            result = {
                "nakshatra": name,
                "nakshatra_index": nakshatra_idx + 1,
                "pada": pada,
                "nature": nature,
                "lord_planet": lord,
                "degree_in_nakshatra": round(offset_in_nakshatra, 4),
            }
            logger.debug(f"Nakshatra: {name} Pada {pada} | degree={degree:.2f} | nature={nature}")
            return result

        except Exception as e:
            logger.error(f"get_nakshatra failed: {e}")
            return {"nakshatra": "Unknown", "pada": 1, "nature": "neutral", "lord_planet": "Unknown"}

    # ------------------------------------------------------------------
    # 3. Tithi
    # ------------------------------------------------------------------
    def get_tithi(self, sun_deg: float, moon_deg: float) -> Dict:
        """
        Calculate lunar tithi from Sun and Moon absolute degrees.
        angle = (moon - sun) % 360; tithi = int(angle/12) + 1
        """
        try:
            sun_deg = float(sun_deg) % 360.0
            moon_deg = float(moon_deg) % 360.0

            angle = (moon_deg - sun_deg) % 360.0
            tithi_number = int(angle / 12.0) + 1
            tithi_number = max(1, min(tithi_number, 30))

            name = TITHI_NAMES[tithi_number - 1]
            nature = TITHI_NATURE.get(tithi_number, "neutral")
            paksha = "Shukla" if tithi_number <= 15 else "Krishna"

            # Add suffix for Krishna paksha tithis
            display_name = f"{paksha} {name}"

            result = {
                "tithi_number": tithi_number,
                "name": display_name,
                "nature": nature,
                "paksha": paksha,
                "sun_moon_angle": round(angle, 4),
            }
            logger.debug(f"Tithi: {display_name} (#{tithi_number}) | nature={nature} | paksha={paksha}")
            return result

        except Exception as e:
            logger.error(f"get_tithi failed: {e}")
            return {"tithi_number": 1, "name": "Unknown", "nature": "neutral", "paksha": "Shukla"}

    # ------------------------------------------------------------------
    # 4. Hora
    # ------------------------------------------------------------------
    def get_hora(self, dt: datetime) -> Dict:
        """
        Determine the planetary hora for a given datetime.
        Day starts at sunrise (~6:00 AM Mumbai). Each hora ~1 hour.
        Day lord determines the first hora; subsequent follow Chaldean order.
        """
        try:
            # Approximate Mumbai sunrise at 6:00 AM
            sunrise_hour = 6
            sunrise_minute = 0
            sunrise = dt.replace(hour=sunrise_hour, minute=sunrise_minute, second=0, microsecond=0)

            # If before sunrise, consider previous day
            if dt.hour < sunrise_hour:
                effective_day = (dt.weekday() - 1) % 7
                time_since_sunrise = (dt - sunrise).total_seconds() + 86400  # add 24h
            else:
                effective_day = dt.weekday()
                time_since_sunrise = (dt - sunrise).total_seconds()

            # Each hora is approximately 1 hour (24 horas per day)
            hora_number = int(time_since_sunrise / 3600) % 24

            # Get day lord
            day_lord = DAY_LORDS.get(effective_day, "Sun")

            # Find day lord's position in Chaldean sequence
            day_lord_idx = HORA_SEQUENCE.index(day_lord) if day_lord in HORA_SEQUENCE else 0

            # Current hora planet: start from day lord, advance by hora_number
            current_idx = (day_lord_idx + hora_number) % 7
            hora_planet = HORA_SEQUENCE[current_idx]

            is_benefic = hora_planet in BENEFIC_PLANETS
            nature = "benefic" if is_benefic else "malefic"
            score_modifier = 4 if is_benefic else -2

            result = {
                "hora_planet": hora_planet,
                "hora_number": hora_number + 1,
                "day_lord": day_lord,
                "nature": nature,
                "is_benefic": is_benefic,
                "score_modifier": score_modifier,
                "effective_day": effective_day,
            }
            logger.debug(f"Hora: {hora_planet} ({nature}) | hora #{hora_number+1} | day_lord={day_lord}")
            return result

        except Exception as e:
            logger.error(f"get_hora failed: {e}")
            return {
                "hora_planet": "Unknown", "nature": "neutral",
                "is_benefic": False, "score_modifier": 0,
            }

    # ------------------------------------------------------------------
    # 5. Planetary aspects
    # ------------------------------------------------------------------
    def get_planetary_aspects(self, positions: Dict) -> List[Dict]:
        """
        Check major aspects between all planet pairs.
        Aspects: Conjunction (<8), Sextile (60+-6), Square (90+-6),
                 Trine (120+-6), Opposition (180+-8).
        """
        try:
            aspects: List[Dict] = []
            planet_names = list(positions.keys())

            for i in range(len(planet_names)):
                for j in range(i + 1, len(planet_names)):
                    p1_name = planet_names[i]
                    p2_name = planet_names[j]
                    p1_deg = positions[p1_name]["degree"]
                    p2_deg = positions[p2_name]["degree"]

                    # Angular separation (shortest arc)
                    diff = abs(p1_deg - p2_deg)
                    if diff > 180:
                        diff = 360 - diff

                    for aspect_name, aspect_info in ASPECT_TYPES.items():
                        target_angle = aspect_info["angle"]
                        orb_limit = aspect_info["orb"]
                        orb = abs(diff - target_angle)

                        if orb <= orb_limit:
                            # Strength: tighter orb = stronger aspect
                            strength = round(1.0 - (orb / orb_limit), 3)
                            market_impact = ASPECT_MARKET_IMPACT.get(aspect_name, "neutral")

                            # Malefic pair amplifies negative impact
                            if p1_name in {"saturn", "mars", "rahu", "ketu"} and \
                               p2_name in {"saturn", "mars", "rahu", "ketu"}:
                                if aspect_name in ("Conjunction", "Square", "Opposition"):
                                    market_impact = "strongly_bearish"

                            aspects.append({
                                "planet1": p1_name,
                                "planet2": p2_name,
                                "type": aspect_name,
                                "orb": round(orb, 4),
                                "market_impact": market_impact,
                                "strength": strength,
                            })

            aspects.sort(key=lambda x: x["strength"], reverse=True)
            logger.info(f"Planetary aspects computed | found={len(aspects)}")
            return aspects

        except Exception as e:
            logger.error(f"get_planetary_aspects failed: {e}")
            return []

    # ------------------------------------------------------------------
    # 6. Yoga calculation
    # ------------------------------------------------------------------
    def get_yoga(self, sun_deg: float, moon_deg: float) -> Dict:
        """
        Calculate Yoga from Sun + Moon longitudes.
        Yoga = (Sun_degree + Moon_degree) / (360/27)
        27 Yogas in Vedic astrology.
        """
        YOGA_NAMES = [
            "Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana",
            "Atiganda", "Sukarma", "Dhriti", "Shula", "Ganda",
            "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra",
            "Siddhi", "Vyatipata", "Variyan", "Parigha", "Shiva",
            "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma",
            "Indra", "Vaidhriti",
        ]
        YOGA_NATURE = {
            "Vishkambha": "bearish", "Priti": "bullish", "Ayushman": "bullish",
            "Saubhagya": "bullish", "Shobhana": "bullish", "Atiganda": "bearish",
            "Sukarma": "bullish", "Dhriti": "bullish", "Shula": "bearish",
            "Ganda": "bearish", "Vriddhi": "bullish", "Dhruva": "bullish",
            "Vyaghata": "bearish", "Harshana": "bullish", "Vajra": "volatile",
            "Siddhi": "bullish", "Vyatipata": "bearish", "Variyan": "bullish",
            "Parigha": "bearish", "Shiva": "bullish", "Siddha": "bullish",
            "Sadhya": "bullish", "Shubha": "bullish", "Shukla": "bullish",
            "Brahma": "bullish", "Indra": "bullish", "Vaidhriti": "bearish",
        }
        try:
            total = (sun_deg + moon_deg) % 360
            yoga_index = int(total / (360 / 27))
            yoga_name = YOGA_NAMES[yoga_index]
            return {
                "name": yoga_name,
                "number": yoga_index + 1,
                "nature": YOGA_NATURE.get(yoga_name, "neutral"),
            }
        except Exception as e:
            logger.error(f"get_yoga failed: {e}")
            return {"name": "--", "number": 0, "nature": "neutral"}

    # ------------------------------------------------------------------
    # 7. Karana calculation
    # ------------------------------------------------------------------
    def get_karana(self, sun_deg: float, moon_deg: float) -> Dict:
        """
        Calculate Karana (half-tithi).
        Each tithi has 2 karanas. 11 karanas total, 4 fixed + 7 rotating.
        """
        KARANA_NAMES = [
            "Bava", "Balava", "Kaulava", "Taitila", "Garaja",
            "Vanija", "Vishti",  # 7 rotating (Vishti = Bhadra = inauspicious)
        ]
        FIXED_KARANAS = ["Shakuni", "Chatushpada", "Naga", "Kimstughna"]
        KARANA_NATURE = {
            "Bava": "bullish", "Balava": "bullish", "Kaulava": "bullish",
            "Taitila": "bullish", "Garaja": "bullish", "Vanija": "bullish",
            "Vishti": "bearish",  # Bhadra karana — inauspicious
            "Shakuni": "bearish", "Chatushpada": "bearish",
            "Naga": "bearish", "Kimstughna": "neutral",
        }
        try:
            angle = (moon_deg - sun_deg) % 360
            karana_number = int(angle / 6)  # 60 karanas in 360 degrees

            if karana_number == 0:
                name = FIXED_KARANAS[0]
            elif karana_number == 57:
                name = FIXED_KARANAS[1]
            elif karana_number == 58:
                name = FIXED_KARANAS[2]
            elif karana_number == 59:
                name = FIXED_KARANAS[3]
            else:
                name = KARANA_NAMES[(karana_number - 1) % 7]

            return {
                "name": name,
                "number": karana_number + 1,
                "nature": KARANA_NATURE.get(name, "neutral"),
                "is_bhadra": name == "Vishti",
            }
        except Exception as e:
            logger.error(f"get_karana failed: {e}")
            return {"name": "--", "number": 0, "nature": "neutral", "is_bhadra": False}

    # ------------------------------------------------------------------
    # 8. Current snapshot
    # ------------------------------------------------------------------
    def get_current_snapshot(self) -> Dict:
        """
        Combine all astro data for the current datetime into a single snapshot.
        """
        try:
            now = datetime.now()
            logger.info(f"Building astro snapshot | dt={now.isoformat()}")

            positions = self.get_planet_positions(now)

            # Moon nakshatra
            moon_deg = positions.get("moon", {}).get("degree", 0.0)
            nakshatra = self.get_nakshatra(moon_deg)

            # Tithi
            sun_deg = positions.get("sun", {}).get("degree", 0.0)
            tithi = self.get_tithi(sun_deg, moon_deg)

            # Hora
            hora = self.get_hora(now)

            # Aspects
            aspects = self.get_planetary_aspects(positions)

            # Yoga
            yoga = self.get_yoga(sun_deg, moon_deg)

            # Karana
            karana = self.get_karana(sun_deg, moon_deg)

            snapshot = {
                "timestamp": now.isoformat(),
                "positions": positions,
                "nakshatra": nakshatra,
                "tithi": tithi,
                "hora": hora,
                "yoga": yoga,
                "karana": karana,
                "aspects": aspects,
                "moon_sign": positions.get("moon", {}).get("sign", "--"),
                "sun_sign": positions.get("sun", {}).get("sign", "--"),
                "meta": {
                    "locale": "Mumbai, IN",
                    "lat": self.MUMBAI_LAT,
                    "lng": self.MUMBAI_LNG,
                    "tz": self.MUMBAI_TZ,
                },
            }

            logger.info(
                f"Astro snapshot ready | nakshatra={nakshatra.get('nakshatra')} "
                f"| tithi={tithi.get('name')} | hora={hora.get('hora_planet')} "
                f"| aspects={len(aspects)}"
            )
            return snapshot

        except Exception as e:
            logger.error(f"get_current_snapshot failed: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}
