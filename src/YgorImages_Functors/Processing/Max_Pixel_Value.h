//Max_Pixel_Value.h.
#pragma once


#include <list>
#include <functional>
#include <limits>
#include <map>
#include <cmath>

#include <any>

#include "YgorMisc.h"
#include "YgorLog.h"
#include "YgorMath.h"
#include "YgorImages.h"


bool CondenseMaxPixel(planar_image_collection<float,double>::images_list_it_t first_img_it,
                      std::list<planar_image_collection<float,double>::images_list_it_t> selected_img_its,
                      std::list<std::reference_wrapper<planar_image_collection<float,double>>>,
                      std::list<std::reference_wrapper<contour_collection<double>>> ccsl, 
                      std::any );

