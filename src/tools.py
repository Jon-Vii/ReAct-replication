"""Example built‑in tools for the ReAct agent.

Feel free to delete these and replace them with real domain‑specific tools.
"""
from __future__ import annotations

import re
import math
from typing import Dict, Callable, Any, Union
import numpy as np

# Astronomy tool specific imports
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.constants import c, G, M_sun, R_sun, L_sun, au, pc, k_B, m_p, m_e, h, hbar, sigma_sb
from astroquery.simbad import Simbad
import lightkurve as lk

__all__ = [
    "search_web", "calculator", "Tool", 
    "get_object_info", "astronomical_calculator",
    "retrieve_kepler_tess_light_curve",
    "operate_on_light_curve"
]

# In-memory cache for light curve objects
_cached_light_curves: Dict[str, lk.LightCurve] = {}
_light_curve_id_counter: int = 0

class Tool:
    """A simple class to wrap a tool function and give it a name."""
    def __init__(self, name: str, function: Callable[..., str], description: str):
        self.name = name
        self.function = function
        self.description = description

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.function(*args, **kwargs)


def search_web(query: str, num_results: int = 3) -> str:
    """Placeholder for a web search tool. 
    In a real scenario, this would use an API like Google Search, Bing, etc.
    For now, it returns a mock result.
    """
    # This is a mock implementation.
    # Replace with actual web search logic if available.
    print(f"Simulating web search for: '{query}' (returning {num_results} results)")
    mock_results = [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]
    return "\n".join(mock_results[:num_results])

# The old calculator function (previously here) is now removed.
# It is replaced by the Calculator class and its instance below.

class Calculator:
    def __init__(self):
        # This regex is used to find simple arithmetic expressions like '1 + 1' or '2 * (3 + 5)'
        # It looks for numbers (integers or floats) and basic operators (+, -, *, /, parentheses).
        # Simplified regex to remove potentially problematic zero-width space.
        self.pattern = re.compile(r"^\s*([\d\.\s\+\-\*\/()]+)\s*$")

    def run(self, query: str) -> str:
        # Sanitize the query to prevent arbitrary code execution if using eval directly
        # For this example, we rely on the regex to only allow safe characters.
        match = self.pattern.fullmatch(query)
        if not match:
            return "Invalid characters in expression. Only numbers, basic operators (+, -, *, /), and parentheses are allowed."
        try:
            # IMPORTANT: eval() can be dangerous if the input is not strictly controlled.
            # The regex above provides some control, but for a production system,
            # a proper expression parser would be much safer.
            return str(eval(match.group(1)))
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error evaluating expression: {type(e).__name__}"

calculator = Calculator() # This is the instance of the Calculator class, intended for export.

def get_object_info(object_name: str = None, ra: float = None, dec: float = None) -> Dict[str, Any]:
    """Looks up information about an astronomical object by its name or coordinates using SIMBAD.

    Args:
        object_name (str, optional): The name of the astronomical object (e.g., 'M31', 'Sirius').
        ra (float, optional): Right Ascension in decimal degrees. Required if object_name is not provided.
        dec (float, optional): Declination in decimal degrees. Required if object_name is not provided.

    Returns:
        Dict[str, Any]: A dictionary containing object information:
            'name': Main identifier in SIMBAD.
            'ra_deg': Right Ascension in decimal degrees.
            'dec_deg': Declination in decimal degrees.
            'ra_hms': Right Ascension in H:M:S format.
            'dec_dms': Declination in D:M:S format.
            'object_types': List of object types (e.g., ['Galaxy', 'LINER']).
            'flux_data': Available flux data (if any).
            'error': Error message if lookup fails or object not found.
    """
    if not object_name and (ra is None or dec is None):
        return {"error": "Either object_name or both ra and dec must be provided."}

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('otype', 'fluxdata(V)','fluxdata(B)','fluxdata(R)','fluxdata(I)','fluxdata(J)','fluxdata(H)','fluxdata(K)')

    try:
        if object_name:
            result_table = custom_simbad.query_object(object_name)
            if result_table is None or len(result_table) == 0:
                return {"error": f"Object '{object_name}' not found in SIMBAD."}
        elif ra is not None and dec is not None:
            coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
            # Query a region; SIMBAD's cone search might return multiple objects.
            # We'll take the first one for simplicity, assuming it's the closest/primary.
            result_table = custom_simbad.query_region(coord, radius='5s') # 5 arcsecond radius
            if result_table is None or len(result_table) == 0:
                return {"error": f"No object found at RA={ra}, Dec={dec} within 5 arcsec radius."}
        else:
            # This case should ideally be caught by the initial check, but as a fallback:
            return {"error": "Insufficient parameters for SIMBAD query."}

        # Process the first result
        obj_info = result_table[0]
        
        # RA and Dec can sometimes be returned as bytes, decode if necessary
        obj_ra_str = obj_info['RA']
        obj_dec_str = obj_info['DEC']
        if isinstance(obj_ra_str, bytes):
            obj_ra_str = obj_ra_str.decode('utf-8')
        if isinstance(obj_dec_str, bytes):
            obj_dec_str = obj_dec_str.decode('utf-8')
        
        coords = SkyCoord(obj_ra_str, obj_dec_str, unit=(u.hourangle, u.deg), frame='icrs')

        flux_data = {}
        for band in ['V', 'B', 'R', 'I', 'J', 'H', 'K']:
            col_name = f'FLUX_{band}'
            if col_name in obj_info.colnames:
                value = obj_info[col_name]
                # Check if the value is the specific numpy masked constant, or None
                if value is np.ma.masked or value is None:
                    continue # Skip masked or None values
                
                try:
                    # Attempt to convert to float. For Astropy quantities, use .value.
                    # For simple numbers or strings convertible to numbers, float() is fine.
                    numeric_value = value.value if hasattr(value, 'value') else value
                    val_float = float(numeric_value)
                    
                    if not math.isnan(val_float):
                        flux_data[band] = val_float
                except (TypeError, ValueError, AttributeError):
                    # This handles cases where conversion to float fails for non-masked, non-None values
                    # (e.g., a string like '---' if SIMBAD returns that for missing flux, or other non-numeric types)
                    # AttributeError can occur if .value is accessed on a type that doesn't have it and isn't directly float-convertible.
                    pass # Skip if not a valid, non-NaN number

        obj_type_str = obj_info['OTYPE']
        if isinstance(obj_type_str, bytes):
            obj_type_str = obj_type_str.decode('utf-8')

        return {
            "name": obj_info['MAIN_ID'].decode('utf-8') if isinstance(obj_info['MAIN_ID'], bytes) else obj_info['MAIN_ID'],
            "ra_deg": coords.ra.deg,
            "dec_deg": coords.dec.deg,
            "ra_hms": coords.ra.to_string(unit=u.hourangle, sep='hms', precision=2),
            "dec_dms": coords.dec.to_string(unit=u.deg, sep='dms', precision=2, alwayssign=True),
            "object_types": obj_type_str.split('|'), # OTYPE might not be pipe-separated, adjust if needed
            "flux_data": flux_data,
            "error": None
        }
    except Exception as e:
        return {"error": f"SIMBAD query failed: {str(e)}"}

def astronomical_calculator(operation: str, **kwargs) -> Dict[str, Any]:
    """Performs various astronomical calculations.

    Args:
        operation (str): The calculation to perform. Supported operations:
            'convert_units': Converts a value from one unit to another.
                Requires: `value` (float), `from_unit` (str), `to_unit` (str).
            'angular_separation': Calculates angular separation between two sky coordinates.
                Requires: `ra1_deg` (float), `dec1_deg` (float), `ra2_deg` (float), `dec2_deg` (float).
            'get_constant': Retrieves the value of an astronomical constant.
                Requires: `constant_name` (str, e.g., 'c', 'G', 'M_sun').

    Returns:
        Dict[str, Any]: A dictionary containing:
            'result': The calculated value or constant's value.
            'unit': The unit of the result (if applicable).
            'error': Error message if calculation fails.
    """
    try:
        if operation == "convert_units":
            value = kwargs.get('value')
            from_unit_str = kwargs.get('from_unit')
            to_unit_str = kwargs.get('to_unit')
            if value is None or from_unit_str is None or to_unit_str is None:
                return {"error": "'convert_units' requires 'value', 'from_unit', and 'to_unit'."}
            
            from_unit = u.Unit(from_unit_str)
            to_unit = u.Unit(to_unit_str)
            quantity = value * from_unit
            result_quantity = quantity.to(to_unit)
            return {"result": result_quantity.value, "unit": str(result_quantity.unit), "error": None}

        elif operation == "angular_separation":
            ra1_deg = kwargs.get('ra1_deg')
            dec1_deg = kwargs.get('dec1_deg')
            ra2_deg = kwargs.get('ra2_deg')
            dec2_deg = kwargs.get('dec2_deg')
            if None in [ra1_deg, dec1_deg, ra2_deg, dec2_deg]:
                return {"error": "'angular_separation' requires 'ra1_deg', 'dec1_deg', 'ra2_deg', 'dec2_deg'."}

            coord1 = SkyCoord(ra1_deg, dec1_deg, unit=(u.deg, u.deg), frame='icrs')
            coord2 = SkyCoord(ra2_deg, dec2_deg, unit=(u.deg, u.deg), frame='icrs')
            separation = coord1.separation(coord2)
            return {"result": separation.arcsecond, "unit": "arcsec", "error": None} # Return in arcseconds for typical use
        
        elif operation == "get_constant":
            constant_name = kwargs.get('constant_name')
            if constant_name is None:
                return {"error": "'get_constant' requires 'constant_name'."}
            
            # A small dictionary to map common names to astropy.constants objects
            constants_map = {
                'c': c, 'speed_of_light': c,
                'G': G, 'gravitational_constant': G,
                'M_sun': M_sun, 'solar_mass': M_sun,
                'R_sun': R_sun, 'solar_radius': R_sun,
                'L_sun': L_sun, 'solar_luminosity': L_sun,
                'au': au, 'astronomical_unit': au,
                'pc': pc, 'parsec': pc,
                'k_B': k_B, 'boltzmann_constant': k_B,
                'm_p': m_p, 'proton_mass': m_p,
                'm_e': m_e, 'electron_mass': m_e,
                'h': h, 'planck_constant': h,
                'hbar': hbar, 'reduced_planck_constant': hbar,
                'sigma_sb': sigma_sb, 'stefan_boltzmann_constant': sigma_sb
            }
            
            if constant_name in constants_map:
                constant = constants_map[constant_name]
                return {"result": constant.value, "unit": str(constant.unit), "error": None}
            else:
                return {"error": f"Constant '{constant_name}' not recognized. Available: {list(constants_map.keys())}"}

        else:
            return {"error": f"Unsupported operation: '{operation}'. Supported: 'convert_units', 'angular_separation', 'get_constant'."}
    
    except u.UnitConversionError as e:
        return {"error": f"Unit conversion error: {str(e)}"}
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}

def retrieve_kepler_tess_light_curve(
    target_identifier: str,
    mission: str = None,
    sector_or_quarter: int = None,
    author: str = None,
    exptime: Union[str, int] = None
) -> Dict[str, Any]:
    """Searches for and downloads a light curve from Kepler, K2, or TESS missions.

    The downloaded light curve object is cached in memory, and a unique ID is returned.
    This ID can be used by other tools to access the cached light curve.

    Args:
        target_identifier (str): The name or identifier of the target (e.g., 'KIC 8462852', 'pi Men', 'TIC 270859903').
        mission (str, optional): The mission to search ('Kepler', 'K2', 'TESS'). 
                                 If None, lightkurve will try to infer or search across missions.
        sector_or_quarter (int, optional): Specific sector (for TESS) or quarter (for Kepler/K2) to search.
        author (str, optional): Specific author of the light curve data product (e.g., 'Kepler', 'SPOC', 'QLP', 'TESS-SPOC').
        exptime (Union[str, int], optional): Exposure time. For TESS, common values are 'short', 'long', 20, 120, 200, 1800 (seconds).
                                            For Kepler, it's typically 'long' (1800s) or 'short' (60s).

    Returns:
        Dict[str, Any]: A dictionary containing:
            'status': 'success' or 'error'.
            'message': A descriptive message.
            'lc_id': (on success) The unique ID for the cached light curve.
            'target': (on success) The target identifier used by lightkurve.
            'mission': (on success) Mission of the downloaded light curve.
            'author': (on success) Author of the data product.
            'exptime_s': (on success) Exposure time in seconds.
            'num_points': (on success) Number of data points in the light curve.
            'search_term': The original target_identifier used for the search.
    """
    global _cached_light_curves, _light_curve_id_counter
    try:
        search_kwargs = {}
        if mission:
            search_kwargs['mission'] = mission
        if sector_or_quarter is not None: # Check for None explicitly because 0 is a valid quarter/sector
            if isinstance(mission, str) and mission.upper() == 'TESS':
                search_kwargs['sector'] = sector_or_quarter
            else: # Kepler, K2, or if mission is None and we guess
                search_kwargs['quarter'] = sector_or_quarter
        if author:
            search_kwargs['author'] = author
        if exptime:
            search_kwargs['exptime'] = exptime

        # Pass target_identifier as the first positional argument
        # and other parameters as keyword arguments.
        search_result = lk.search_lightcurve(target_identifier, **search_kwargs)

        if not search_result:
            return {
                "status": "error",
                "message": f"No light curves found for target '{target_identifier}' with specified parameters.",
                "search_term": target_identifier
            }

        # For simplicity, download the first available light curve from the search result
        # More sophisticated selection could be added later (e.g., latest sector, specific author)
        lc = search_result[0].download()
        
        if lc is None:
             return {
                "status": "error",
                "message": f"Failed to download light curve for '{target_identifier}' from search results.",
                "search_term": target_identifier
            }

        _light_curve_id_counter += 1
        lc_id = f"lc_{_light_curve_id_counter}"
        _cached_light_curves[lc_id] = lc

        # Try to get exposure time in seconds, might be a Quantity or a value
        try:
            exp_time_s = lc.exptime.to(u.s).value if hasattr(lc.exptime, 'to') else float(lc.exptime)
        except (AttributeError, TypeError, ValueError):
            exp_time_s = 'unknown'

        return {
            "status": "success",
            "message": f"Light curve for '{lc.label}' downloaded and cached with ID: {lc_id}",
            "lc_id": lc_id,
            "target": lc.label, # label often contains the best identifier
            "mission": lc.mission.upper() if lc.mission else 'Unknown',
            "author": lc.author.upper() if lc.author else 'Unknown',
            "exptime_s": exp_time_s,
            "num_points": len(lc.flux),
            "search_term": target_identifier
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving light curve for '{target_identifier}': {str(e)}",
            "search_term": target_identifier
        }

def operate_on_light_curve(lc_id: str, operation: str, **kwargs) -> Dict[str, Any]:
    """Performs operations on a cached light curve.

    Args:
        lc_id (str): The ID of the cached light curve (obtained from retrieve_kepler_tess_light_curve).
        operation (str): The operation to perform. Supported operations:
            'get_summary': Returns basic metadata about the light curve.
            'normalize': Normalizes the light curve (divides by median flux). Updates cache.
            'flatten': Flattens the light curve using Savitzky-Golay filter. Updates cache.
                         Optional kwargs: window_length (int, default 51), polyorder (int, default 2).
            'remove_outliers': Removes outliers using sigma clipping. Updates cache.
                               Optional kwargs: sigma (float, default 5.0).
            'calculate_periodogram': Calculates a Lomb-Scargle periodogram.
                                     Optional kwargs: minimum_period (float), maximum_period (float), method (str, default 'lombscargle').
                                     Returns period_at_max_power, max_power.
        **kwargs: Additional arguments specific to the operation.

    Returns:
        Dict[str, Any]: A dictionary containing the status and result of the operation.
    """
    global _cached_light_curves

    if lc_id not in _cached_light_curves:
        return {"status": "error", "message": f"Light curve with ID '{lc_id}' not found in cache."}

    lc = _cached_light_curves[lc_id]
    original_lc_label = lc.label # For messages

    try:
        if operation == "get_summary":
            # Try to get exposure time in seconds, might be a Quantity or a value
            try:
                exp_time_s = lc.exptime.to(u.s).value if hasattr(lc.exptime, 'to') else float(lc.exptime)
            except (AttributeError, TypeError, ValueError):
                exp_time_s = 'unknown'
            
            # Calculate time span robustly
            time_span_val = 'Unknown'
            if lc.time is not None and len(lc.time) > 0:
                try:
                    time_span_val = (lc.time.max() - lc.time.min()).to_value(u.day)
                except Exception: # Catch any exception during time span calculation
                    pass # Keep 'Unknown' if calculation fails
            
            return {
                "status": "success",
                "lc_id": lc_id,
                "target": lc.label,
                "mission": lc.mission.upper() if lc.mission else 'Unknown',
                "author": lc.author.upper() if lc.author else 'Unknown',
                "time_span_days": time_span_val,
                "num_points": len(lc.flux),
                "mean_flux": lc.flux.mean().value if hasattr(lc.flux, 'mean') else 'Unknown',
                "median_flux": lc.flux.median().value if hasattr(lc.flux, 'median') else 'Unknown',
                "exptime_s": exp_time_s,
                "operation": operation
            }

        elif operation == "normalize":
            normalized_lc = lc.normalize()
            _cached_light_curves[lc_id] = normalized_lc # Update cache
            return {
                "status": "success", 
                "lc_id": lc_id, 
                "message": f"Light curve '{original_lc_label}' (ID: {lc_id}) normalized.",
                "operation": operation
            }

        elif operation == "flatten":
            window_length = kwargs.get("window_length", 51) # Default from lightkurve
            polyorder = kwargs.get("polyorder", 2)       # Default from lightkurve
            flattened_lc = lc.flatten(window_length=window_length, polyorder=polyorder)
            _cached_light_curves[lc_id] = flattened_lc # Update cache
            return {
                "status": "success", 
                "lc_id": lc_id, 
                "message": f"Light curve '{original_lc_label}' (ID: {lc_id}) flattened (window={window_length}, polyorder={polyorder}).",
                "operation": operation
            }

        elif operation == "remove_outliers":
            sigma = kwargs.get("sigma", 5.0)
            cleaned_lc = lc.remove_outliers(sigma=sigma)
            _cached_light_curves[lc_id] = cleaned_lc # Update cache
            return {
                "status": "success", 
                "lc_id": lc_id, 
                "message": f"Outliers removed from '{original_lc_label}' (ID: {lc_id}) using sigma={sigma}. Original points: {len(lc.flux)}, New points: {len(cleaned_lc.flux)}.",
                "operation": operation,
                "points_removed": len(lc.flux) - len(cleaned_lc.flux)
            }

        elif operation == "calculate_periodogram":
            method = kwargs.get("method", "lombscargle") # 'lombscargle' or 'bls' for BoxLeastSquares
            # Lightkurve's to_periodogram takes method as first arg, then kwargs for that method
            periodogram_kwargs = {k: v for k, v in kwargs.items() if k not in ['method']}
            pg = lc.to_periodogram(method=method, **periodogram_kwargs)
            
            result_data = {
                "period_at_max_power_days": pg.period_at_max_power.to_value(u.day),
                "max_power": pg.max_power.value if hasattr(pg.max_power, 'value') else pg.max_power, # Handle unitless power for some methods
                "method": method,
            }
            if method.lower() == 'bls': # BLS specific results
                result_data["bls_depth"] = pg.depth_at_max_power
                result_data["bls_duration_days"] = pg.duration_at_max_power.to_value(u.day)
                result_data["bls_transit_time_bkjd"] = pg.transit_time_at_max_power.value # BKJD or BTJD
                # Add more BLS results if needed
            
            return {
                "status": "success",
                "lc_id": lc_id,
                "message": f"Periodogram ({method}) calculated for '{original_lc_label}' (ID: {lc_id}).",
                "operation": operation,
                "results": result_data
            }

        else:
            return {"status": "error", "message": f"Unsupported operation: '{operation}' for light curve ID '{lc_id}'."}

    except Exception as e:
        return {
            "status": "error", 
            "lc_id": lc_id,
            "message": f"Error performing operation '{operation}' on light curve '{original_lc_label}' (ID: {lc_id}): {str(e)}",
            "operation": operation
        }


# Example usage (for testing, can be removed or commented out)
if __name__ == "__main__":
    # Test get_object_info
    print("--- Testing get_object_info ---")
    m31_info = get_object_info(object_name="M31")
    print(f"M31 Info: {m31_info}")

    kic8462852_info = get_object_info(object_name="KIC 8462852") # Added test for KIC 8462852
    print(f"KIC 8462852 Info: {kic8462852_info}")

    # Test by coordinates (example: M31's approx coords)
    m31_coords_info = get_object_info(ra=10.6847, dec=41.2690)
    print(f"M31 Coordinates Info: {m31_coords_info}")

    # Test astronomical_calculator
    conversion = astronomical_calculator("convert_units", value=10, from_unit="pc", to_unit="lyr")
    print("\n10 pc to lyr:", conversion)

    conversion_fail = astronomical_calculator("convert_units", value=10, from_unit="pc", to_unit="kg") # Invalid conversion
    print("\n10 pc to kg (fail test):", conversion_fail)

    separation = astronomical_calculator("angular_separation", ra1_deg=0, dec1_deg=0, ra2_deg=1, dec2_deg=0)
    print("\nAngular separation:", separation)

    light_speed = astronomical_calculator("get_constant", constant_name="c")
    print("\nSpeed of light:", light_speed)

    unknown_const = astronomical_calculator("get_constant", constant_name="xyz")
    print("\nUnknown constant:", unknown_const)

    unknown_op = astronomical_calculator("fly_to_moon", speed="fast")
    print("\nUnknown operation:", unknown_op)

    # Test retrieve_kepler_tess_light_curve
    print("\n--- Light Curve Tests ---")
    # Example 1: A known Kepler target
    kepler_lc_info = retrieve_kepler_tess_light_curve(target_identifier="KIC 8462852", mission="Kepler")
    print("Kepler Light Curve Info:", kepler_lc_info)
    if kepler_lc_info.get("status") == "success":
        print(f"Cached LC object: {_cached_light_curves.get(kepler_lc_info['lc_id'])}")

    # Example 2: A known TESS target (e.g., pi Men, TIC 261136679)
    # Note: TESS data can be large and searches might return many sectors.
    # This will download the first one found.
    tess_lc_info = retrieve_kepler_tess_light_curve(target_identifier="pi Men", mission="TESS", exptime="short")
    print("\nTESS Light Curve Info (pi Men, short exptime):", tess_lc_info)
    if tess_lc_info.get("status") == "success":
        print(f"Cached LC object: {_cached_light_curves.get(tess_lc_info['lc_id'])}")

    # Example 3: Target that might not have data or typo
    no_data_lc_info = retrieve_kepler_tess_light_curve(target_identifier="NonExistentStar123XYZ")
    print("\nNo Data Light Curve Info:", no_data_lc_info)

    print("\n--- Light Curve Operations Tests ---")
    # Use the Kepler LC from previous test if successful
    if kepler_lc_info.get("status") == "success":
        lc_id_kepler = kepler_lc_info['lc_id']
        print(f"\nOperating on Kepler LC ID: {lc_id_kepler}")

        summary = operate_on_light_curve(lc_id_kepler, "get_summary")
        print("Summary:", summary)

        normalized = operate_on_light_curve(lc_id_kepler, "normalize")
        print("Normalize Result:", normalized)
        # Verify cache updated (optional check)
        # if normalized.get("status") == "success":
        #     print("Normalized LC mean flux:", _cached_light_curves[lc_id_kepler].flux.mean())

        flattened = operate_on_light_curve(lc_id_kepler, "flatten", window_length=101, polyorder=3)
        print("Flatten Result:", flattened)

        periodogram_ls = operate_on_light_curve(lc_id_kepler, "calculate_periodogram", minimum_period=0.5, maximum_period=100)
        print("Lomb-Scargle Periodogram Result:", periodogram_ls)

        # Example of removing outliers - this will modify the LC for subsequent ops if successful
        # Could re-retrieve if a clean version is needed for other tests.
        outliers_removed = operate_on_light_curve(lc_id_kepler, "remove_outliers", sigma=3.0)
        print("Remove Outliers Result:", outliers_removed)
        if outliers_removed.get("status") == "success":
            summary_after_clean = operate_on_light_curve(lc_id_kepler, "get_summary")
            print("Summary after outlier removal:", summary_after_clean)
        
        # Test BLS periodogram
        # Note: KIC 8462852 (Boyajian's Star) is complex, BLS might find something or not, depending on data segment
        # For BLS, often good to provide an estimated period range or specific durations to search if known
        # For a general test, we'll let it search broadly. This can be slow.
        # print("Calculating BLS periodogram (can be slow)...")
        # periodogram_bls = operate_on_light_curve(lc_id_kepler, "calculate_periodogram", method='bls', 
        #                                        duration=[0.05, 0.1, 0.2]) # Example durations in days
        # print("BLS Periodogram Result:", periodogram_bls)

    # Test with a non-existent lc_id
    bad_lc_op = operate_on_light_curve("lc_invalid_id", "get_summary")
    print("\nBad LC ID Operation:", bad_lc_op)

    # Test with an unsupported operation
    if kepler_lc_info.get("status") == "success":
        lc_id_kepler = kepler_lc_info['lc_id']
        unsupported_op = operate_on_light_curve(lc_id_kepler, "magically_find_aliens")
        print("Unsupported Operation Result:", unsupported_op)
