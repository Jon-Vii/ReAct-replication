import os

# Assuming 'src' is treated as a package when run with 'python -m src.demo'
# or that 'src' path is correctly handled by the execution environment.
from .react_agent import ReactAgent
from .tools import (
    Tool, 
    search_web, 
    calculator, # This is the Calculator class instance
    get_object_info,
    astronomical_calculator,
    retrieve_kepler_tess_light_curve,
    operate_on_light_curve
)

def main():
    # Ensure the API key is set (export OPENAI_API_KEY=sk‑...)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    # Pass tools as a list of Tool objects
    tools_list = [
        Tool(name="SearchWeb", 
             function=search_web, 
             description="Searches the web for information. Input is the search query string."),
        Tool(name="Calculator", 
             function=calculator.run, # Access the run method of the Calculator instance
             description="Evaluates basic arithmetic expressions. Input is the math expression string. Example: '2 * (3 + 5)'"),
        Tool(name="GetObjectInfo", 
             function=get_object_info, 
             description="Looks up information about an astronomical object by its name or coordinates using SIMBAD. Input: object_name (str), or ra (float) and dec (float). Returns a dictionary with object details or an error."),
        Tool(name="AstronomicalCalculator", 
             function=astronomical_calculator, 
             description="Performs astronomical calculations: 'convert_units' (value, from_unit, to_unit), 'angular_separation' (ra1_deg, dec1_deg, ra2_deg, dec2_deg), 'get_constant' (constant_name). Input: operation (str) and relevant keyword arguments. Returns a dictionary with the result or an error."),
        Tool(name="RetrieveLightCurve", 
             function=retrieve_kepler_tess_light_curve, 
             description="Searches for and downloads a light curve from Kepler, K2, or TESS for a target. Caches the light curve and returns a unique 'lc_id'. Input: target_identifier (str). Optional inputs: mission (str), sector_or_quarter (int), author (str), exptime (str or int). Returns a dictionary with status, lc_id, and metadata or an error."),
        Tool(name="OperateOnLightCurve", 
             function=operate_on_light_curve, 
             description="Performs operations on a cached light curve using its 'lc_id'. Supported operations: 'get_summary', 'normalize', 'flatten', 'remove_outliers', 'calculate_periodogram'. Input: lc_id (str), operation (str), and optional keyword arguments for the operation. Returns a dictionary with status and results or an error.")
    ]
    # Increased max_turns for a potentially complex multi-step query
    # Assuming the ReactAgent class takes 'model', 'tools', 'max_turns', 'verbose' as parameters.
    # Please verify these parameter names with your ReactAgent class definition.
    agent = ReactAgent(llm_model="gpt-4o", tools=tools_list, max_turns=15) 

    prompt = """I'm researching the star KIC 8462852, also known as Boyajian's Star.
1. First, can you get me its basic astronomical information, like its coordinates and object type?
2. Then, retrieve its Kepler light curve data.
3. Once you have the light curve, normalize it and then calculate a Lomb-Scargle periodogram. Report the period with the highest power and the value of that power.
4. Finally, what is the value of the astronomical unit (au) in meters?"""
    
    answer = agent.run(prompt)
    print("Final answer:", answer)


if __name__ == "__main__":
    main()
