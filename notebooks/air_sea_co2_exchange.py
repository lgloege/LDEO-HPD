# ===============================================================
# Functions to calculate air-sea CO2 exchange.
# uses the Wanninkho2 (1992) parameterization
# and follows best practices outlined in Dickson et al. (2007)
# 
# Includes
# ------------
# solubility() -- Weiss (1974)
# water_vapor_pressure() -- Dickson et al (2007) or Weiss and Price (1980)
# pco2_moist_atm() -- Dickson et al (2007)
# schmidt_number() -- Wanninkhof (1992)
# gas_transfer_velocity() --Wanninkhof (1992)
# calculate_fgco2() -- Wanninkhof (1992)
# flow_rate 
#
# Author: L. Gloege
# Updated November 15 2019
#
# ===============================================================
import numpy as np

#################################################################
#
# calculates solubility of CO2 in sea water
#
#################################################################
def solubility(T_kelvin=None, S=None):
    '''
    solubility of CO2
    
    inputs
    =======
    T_kelvin : temperature [kelvin]
    S : salinity [psu]
    
    output
    ========
    sol : solbuilty [mol/kg/uatm]
    
    references
    =========
    Weiss (1974) [for solubility of CO2]
        Carbon dioxide in water and seawater: the solubility of a non-ideal gas
    
    test case
    ========
    T=20degC S=20psu Ko=0.03515 mol/kg/atm [from table III in Weiss 74]
    '''
    # Paramters in Table 1 of Weiss (1974)
    # units : mol/kg/atm
    A1 = -60.2409
    A2 = 93.4517
    A3 = 23.3585
    B1 = 0.023517
    B2 = -0.023656
    B3 = 0.0047036
    
    # Equation 12 in Weiss (1974)
    ln_Ko = A1 + A2*(100/T_kelvin) + A3*np.log(T_kelvin/100) + \
            S*(B1 + B2*(T_kelvin/100) + B3*(T_kelvin/100)**2)
    
    # Solbuilty in units mol/kg/uatm
    # the conversion factor at end converts
    # om mol/kg/atm to mol/kg/uatm
    Ko = np.exp(ln_Ko) * 1.0E-6

    return Ko


#################################################################
#
# calculates water vapor pressure
#
#################################################################
def water_vapor_pressure(T_kelvin=None, S=None, weiss_price=False):
    '''
    water vapor pressure over sea water.
    
    ps_sw = ps * exp(-0.018*phi*MT)
    
    where ps is the vapor pressure of pure water
    phi is the osmotic coefficient of sea wtaer
    MT is the total molality of seawter
    
    inputs
    ===========
    T_kelvin    : temperature [kelvin]
    S           : salinity [psu]
    weiss_price : toggle to use Wwiss and Price 1980 parameterization
    
    output
    ===========
    ps_sw : saturation vapor pressure for sea water [atm]
    
    references
    ============
    Dickson et al (2007) [for saturation vapor pressure]
        Guide to best practices for Ocean CO2 measurements, Chapter 5 section 3.2
    Weiss, R. F., and B. A. Price. (1980) 
        "Nitrous oxide solubility in water and seawater." see equation 10
    Sarmiento and Gruber
        Ocean biogeochmiecal dynamics, Ch3 see panel 3.2.1
        [NOTE: equation (8) is for p^H2O, not p^H2O/P]
        
    Test value
    ============
    T_kelvin=273.15+25, S=35, P_sat_vapor=3.1106 kPa (0.030699235134468292 atm)
    '''

    #============================================================
    # This section calculates vapor pressure of pure water
    # following Wagner and Pruß (2002) and Dickson et al. (2007)
    # see Dickson et al (2007) Chapter 5 section 3.1
    # Test Case: T=298.15K, ps=3.1698 kPa (0.030699235134468292atm)
    #============================================================
    # Following Dickson et al (2007) Chapter 5 section 3.1
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502

    # Tc - units of Kelvin
    # pc - units MPa
    # See Dickson et al (2007) Chapter 5  section 3.1 equation 1
    Tc = 647.096 
    pc = 22.064
    g = (1-(T_kelvin/Tc)) 

    # pure water saturation vapor pressure, Wagner and Pruß (2002)
    # See Dickson et al (2007) Chapter 5  section 3.1 equation 1
    ln_ps_pc = (Tc/T_kelvin) * (a1*g + 
                         a2*g**(1.5) + 
                         a3*g**(3) + 
                         a4*g**(3.5) + 
                         a5*g**(4) + 
                         a6*g**(7.5))

    # varpor pressure of pure water in units of Mpa
    ps = pc * np.exp(ln_ps_pc)

    #============================================================
    # Total molality
    # see Dickson et al. (2007) Ch.5 sec.3.2 eqn.3
    #============================================================
    # total molality 
    MT = (31.998*S) / (1000 - 1.005*S)

    #============================================================
    # osmotic coefficient at 25C by Millero (1974)
    # see Dickson et al. (2007) Ch.5 sec.3.2 eqn.4
    #============================================================
    # osmotic coefficient at 25C by Millero (1974)
    b1 = 0.90799 
    b2 = -0.08992
    b3 = 0.18458
    b4 = -0.07395
    b5 = -0.00221
    phi = b1 + b2*(0.5*MT) + b3*(0.5*MT)**2 + b4*(0.5*MT)**3 + b5*(0.5*MT)**4

    #============================================================
    # Vapor pressure for sea water
    # See Dickson et al. (2007) Ch.5, sec.3.2 eqn.2
    # Test Case: T=25C, S=35, ps_sw=3.1106kPa
    # Here I convert to atm 
    # 10**6 converts from Mpa to Pa 
    # 101325 converts from Pa to atm
    #============================================================
    if weiss_price:
        # units in atm
        ps_sw = np.exp(24.4543-67.4509*(100/T_kelvin)-4.8489*np.log(T_kelvin/100)-0.000544*S)
    else:
        # units in atm
        # Note the conversions applied at the end
        ps_sw = (ps * np.exp(-0.018*phi*MT) * 10**6) / 101325
        
    return ps_sw


#################################################################
#
# calculates pco2 in moist air
#
#################################################################
def pco2_moist_atm(T_kelvin=None, S=None, P=None, xCO2=None, weiss_price=False):
    '''
    atmospheric pco2 corrected for vapor pressure
    
    inputs
    ========
    T_kelvin    : temperature [kelvin]
    S           : salinity [psu]
    P           : pressure [atm]
    xCO2        : atmospheric pco2 [ppmv]
    weiss_price : toggle to use Weiss and Price 1980 parameterization
    
    outpus
    ========
    P_moist : partial pressure of co2 in moist air [uatm]
    
    references
    ==========
    Dickson et al (2007) [for saturation vapor pressure]
        Guide to best practices for Ocean CO2 measurements, Chapter 5 section 3.2
    '''
    # vapor pressure
    pH2O = water_vapor_pressure(T_kelvin, S, weiss_price=weiss_price)
    
    # Partial pressure of CO2 in moist air, 
    # Dickson et al. (2007), SOP 4 sec. 8.3
    P_moist = xCO2 * (P - pH2O) 
    
    return P_moist


#################################################################
#
# calculates schmidt number
#
#################################################################
def schmidt_number(T=None):
    '''
    schmidt number : ratio of momentum diffusivity and mass diffusivity
    
    Sc = v/D = Mu/(rho+D)

    where v is the kinematic viscosity of the water
    D is the mass diffusivity, rho is density, and mu is the viscosity.

    characterize fluid flows where there are simultaneous 
    momentum and mass diffusion convection processes
    
    Schmidt numbers are used to estimate the gas transfer velocity.
    
    inputs
    ========
    T : temperature in DegC
    
    outpus
    ========
    Sc : schmidt number 
    
    references
    ==========
    Wanninkhof (1992) 
        Relationship between wind speed and gas exchange over the ocean   
    '''
    # Dimensionless Schmidt number (Sc)
    # References Wanninkhof (1992) Tabl3 A1
    A = 2073.1 
    B = 125.62 
    C = 3.6276 
    D = 0.043219
    
    # Schmidt number [dimensionless]
    Sc = A - B*T + C*T**2 - D*T**3
    
    return Sc


#################################################################
#
# calculates gas transfer velocity
#
#################################################################
def gas_transfer_velocity(T=None, u_mean=None, u_var=None, scale_factor=0.27):
    '''
    Wanninkhof 1992 gas trasnfer velocity 
    kw = scale_factor * (Sc/660)**(-0.5) * (u**2)
    
    inputs
    =========
    T : temperature [degC]
    u_mean : mean wind speed [m/s]
    u_var : variance of wind speed [m2/s2]
    
    outputs
    =========
    kw : gas transfer velocity [cm/hr]
    
    dependencies
    ===========
    You need the schmidt_number() function
    Don't have it? everything is in the code, just uncomment it.
    or look in Wannikhof 1992
    
    references
    ==========
    Wanninkhof (1992) [for gas transfer and Schmidt number]
        Relationship between wind speed and gas exchange over the ocean   
    Sweeney et al. (2007) [for gas transfer scale factor]
        Constraining global air‐sea gas exchange for CO2 with recent bomb 14C measurements
    '''
    # ==================================================================
    # 3. Gas transfer velocity
    #    Units : cm/hr
    #    Reference : Wanninkhof (1992)
    #                Sweeney et al. (2007)
    #                per.comm with P. Landschutzer Feb. 19 2019
    # ==================================================================
    # Dimensionless Schmidt number (Sc)
    # References Wanninkhof (1992) Table.3 A1
    #A,B,C,D = 2073.1, 125.62, 3.6276, 0.043219
    #Sc = A - B*T + C*T**2 - D*T**3
    Sc = schmidt_number(T=T)
    
    # Gas transfer velocity
    # 660 is the Schmidt number of CO2 in seawater at 20C
    # References :  Wanninkhof (1992), Sweeney et al. (2007) scale factor
    kw = scale_factor * (Sc/660)**(-0.5) * (u_mean**2 + u_var)
    
    return kw


#################################################################
#
# calculates air-sea CO2 exchange
#
#################################################################
def calculate_fgco2(T=None, 
                    S=None, 
                    P=None,
                    u_mean=None, 
                    u_var=None, 
                    xCO2=None,
                    pCO2_sw = None,
                    iceFrac = None,
                    scale_factor=0.27):
    '''
    Air-sea CO2 exchange 
    
    fgco2 = k * Sol * (1 - iceFrac) * (spco2_sw - spco2_air)
    
    where, k is the gas transfer velocity, Sol is the solubility
    iceFrac is the ice fraction spco2_sw is the surface pCO2 of sea water
    and spco2_air is the moist air pCO2. 
    
    Inputs
    ============
    T : Temperature [degC]
    S : Salinity  [parts per thousand]
    P : Atmospheric pressure at 10m [atm]
    u_mean  : mean wind speed [m/s]
    u_var   : variance of wind speed m/s averaged monthly [m2/s2]
    xcO2    : atmoapheric mixing ratio of CO2 [ppm]
    pCO2_sw : CO2 partial pressure of seawter [uatm]
    scale_factor : gas transfer scale factor (default=0.27)
    
    Output
    ============
    fgco2 = air-sea co2 flux [molC/m2/yr]
    
    Dependencies
    ===========
    water_vapor_pressure() -- calculated water vapor pressure
    pco2_moist_atm() -- calcualted atmospheric pCO2 in moist are
    solubility() -- calculates solubility of CO2 in sea water
    gas_transfer_velocity() -- calculates gas transfer vel (assume U**2 relation)
   
    References
    ============
    Weiss (1974) [for solubility of CO2]
        Carbon dioxide in water and seawater: the solubility of a non-ideal gas
    Weiss and Price (1980) [for saturation vapor pressure, not used here]
        Nitrous oxide solubility in water and seawater     
    Dickson et al (2007) [for saturation vapor pressure]
        Guide to best practices for Ocean CO2 measurements, Chapter 5 section 3.2
    Wanninkhof (1992) [for gas transfer and Schmidt number]
        Relationship between wind speed and gas exchange over the ocean   
    Sweeney et al. (2007) [for gas transfer scale factor]
        Constraining global air‐sea gas exchange for CO2 with recent bomb 14C measurements
    Sarmiento and Gruber (2007) [for overview and discussion of CO2 flux]
        Ocean biogeochmiecal dynsmics, Ch3 see panel 3.2.1, and panel 3.3.1
    
    Notes
    ============
    - Need to add wind speed variance if using Wanninkhof (1992)
      If we don't, then we are underestimating the gas-transfer velocity
      U2 = (U_mean)^2 + (U_prime)^2, (U_prime)^2 is the variance
      See Sarmiento and Gruber (2007) Ch.3. panel 3.3.1 for 
    - U_prime^2 can be calculated using ERAinterim 6 hourly output
    - functions used : ps_sw(), solubility(), gas_transfer_velocity()
    
    '''
    # ==================================================================
    # 0. temperature 
    #    Units : K
    # ==================================================================
    T_kelvin = T + 273.15
    
    # ==================================================================
    # 1. partial pressure of CO2 in moist air
    #    Units : uatm
    # ==================================================================        
    P_moist = pco2_moist_atm(T_kelvin=T_kelvin, S=S, P=P, xCO2=xCO2)
    
    # ==================================================================
    # 2. Solubility of CO2
    #    Units : mol/kg/uatm
    # ==================================================================  
    Sol = solubility(T_kelvin=T_kelvin, S=S)
    
    # ==================================================================
    # 3. Gas transfer velocity. Note temperature here needs be degC
    #    Units : cm/hr
    # ==================================================================
    kw = gas_transfer_velocity(T=T, u_mean=u_mean, u_var=u_var, scale_factor=scale_factor)

    # ================================================
    # 4. air-sea CO2 exchange
    #    Units : mol/m2/yr
    # ================================================  
    # Convert from cm*mol/hr/kg to mol/m2/yr
    conversion = (1000/100)*365*24
    
    # Air-sea CO2 flux 
    fgco2 = kw * Sol * (1 - iceFrac) * (pCO2_sw - P_moist) * conversion
    
    return fgco2


#################################################################
#
# calculates flow rate
#
#################################################################
def flow_rate(da=None, area=None):
    '''
    calculate CO2 flow rate: flow_rate = Flux * Area
    
    Input
    =======
    da   : CO2 flux [molC/m2/yr]
    area : grid cell area [m2]
    
    Output
    =======
    flow_rate : flow rate of CO2 [PgC/yr]
    
    References
    ========
    '''
    
    # conversion from molC to PgC
    conversion = 12 * (10**(-15))
                       
    # flux [molC / m2 /yr] converted to [PgC /m2/ yr]
    flux = da*conversion

    # CO2 flow rate [PgC / yr]
    flow_rate = flux * area

    return flow_rate