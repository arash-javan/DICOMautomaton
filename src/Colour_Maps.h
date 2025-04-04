//Colour_Maps.h.

#pragma once

#include <optional>
#include <string>
#include <cstdint>


struct ClampedColourRGB {
    double R; // within [0,1].
    double G; // within [0,1].
    double B; // within [0,1].
};


//These functions take a clamped input in [0,1] and map it to a colour specified in terms of R,G,B all within [0,1].

ClampedColourRGB ColourMap_Linear(double y);

ClampedColourRGB ColourMap_Viridis(double y);
ClampedColourRGB ColourMap_Magma(double y);
ClampedColourRGB ColourMap_Inferno(double y);
ClampedColourRGB ColourMap_Plasma(double y);

ClampedColourRGB ColourMap_Jet(double y);

ClampedColourRGB ColourMap_MorelandBlueRed(double y);
ClampedColourRGB ColourMap_MorelandBlackBody(double y);
ClampedColourRGB ColourMap_MorelandExtendedBlackBody(double y);

ClampedColourRGB ColourMap_KRC(double y);
ClampedColourRGB ColourMap_ExtendedKRC(double y);

ClampedColourRGB ColourMap_Kovesi_LinKRYW_5_100_c64(double y);
ClampedColourRGB ColourMap_Kovesi_LinKRYW_0_100_c71(double y);

ClampedColourRGB ColourMap_Kovesi_Cyclic_mygbm_30_95_c78(double y);

ClampedColourRGB ColourMap_LANL_OliveGreen_to_Blue(double y);

ClampedColourRGB ColourMap_YgorIncandescent(double y);

ClampedColourRGB ColourMap_Composite_50_90_107_110(double y);
ClampedColourRGB ColourMap_Composite_50_90_100_107_110(double y);

ClampedColourRGB ColourMap_DICOM_Hot_Iron(double y);
ClampedColourRGB ColourMap_DICOM_PET(double y);
ClampedColourRGB ColourMap_DICOM_Hot_Metal_Blue(double y);
ClampedColourRGB ColourMap_DICOM_PET_20_Step(double y);

//This function takes a named colour and map it to a colour specified in terms of R,G,B all within [0,1].
std::optional<ClampedColourRGB> Colour_from_name(const std::string& n);

//These functions cycle through colours.
ClampedColourRGB Colour_cycle_max_contrast_20(int32_t &i);

