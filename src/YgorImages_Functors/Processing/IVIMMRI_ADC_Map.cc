
#include <cstdlib>
#include <array>
#include <cmath>
#include <any>
#include <optional>
#include <functional>
#include <limits>
#include <list>

#include "../ConvenienceRoutines.h"
#include "YgorImages.h"
#include "YgorMath.h"
#include "YgorMisc.h"
#include "YgorLog.h"
#include "YgorStats.h"       //Needed for Stats:: namespace.

bool IVIMMRIADCMap(planar_image_collection<float,double>::images_list_it_t first_img_it,
                   std::list<planar_image_collection<float,double>::images_list_it_t> selected_img_its,
                   std::list<std::reference_wrapper<planar_image_collection<float,double>>>,
                   std::list<std::reference_wrapper<contour_collection<double>>>, 
                   std::any ){

    //This routine computes an ADC from a series of IVIM images by fitting linearized diffusion b-values.
    const bool InhibitSort = true; //Disable continuous sorting (defer to single sort later) to speed up data ingress.

    //Make a 'working' image which we can edit. Start by duplicating the first image.
    planar_image<float,double> working;
    working = *first_img_it;

    //Paint all pixels black.
    working.fill_pixels(static_cast<float>(0));

    //Record the min and max actual pixel values for windowing purposes.
    Stats::Running_MinMax<float> minmax_pixel;

    //Loop over the rois, rows, columns, channels, and finally any selected images (if applicable).
    for(auto row = 0; row < first_img_it->rows; ++row){
        for(auto col = 0; col < first_img_it->columns; ++col){
            for(auto chan = 0; chan < first_img_it->channels; ++chan){

                //Cycle over the grouped images (diffusion b-values).
                // Harvest the time course or any other voxel-specific numbers.
                samples_1D<double> channel_bval_course;
                channel_bval_course.uncertainties_known_to_be_independent_and_random = true;
                for(auto & img_it : selected_img_its){
                    //Collect the datum of voxels and nearby voxels for an average.
                    std::list<double> in_pixs;
                    auto boxr = 1; //The inclusive 'radius' of the square box to use to average nearby pixels.
                    for(auto lrow = (row-boxr); lrow <= (row+boxr); ++lrow){
                        for(auto lcol = (col-boxr); lcol <= (col+boxr); ++lcol){
                            //Check if the coordinates are legal.
                            if( isininc(0,lrow,img_it->rows-1) 
                            &&  isininc(0,lcol,img_it->columns-1) ){ 
                                const auto val = static_cast<double>(img_it->value(lrow, lcol, chan));
                                in_pixs.push_back(val);
                            }
                        }
                    }
                    if(in_pixs.size() < 3) continue; //Too few to bother with.
                    const auto avg_val = Stats::Mean(in_pixs);
                    const auto avg_val_sigma = std::sqrt(Stats::Unbiased_Var_Est(in_pixs))/std::sqrt(1.0*in_pixs.size());

                    auto bval = img_it->GetMetadataValueAs<double>("Diffusion_bValue");
                    if(!bval) YLOGERR("Image missing diffusion b-value. Cannot continue");
                    channel_bval_course.push_back(bval.value(), 0.0, avg_val, avg_val_sigma, InhibitSort);
                }
                channel_bval_course.stable_sort();
                if(channel_bval_course.empty()) continue;
   
                //Perform regression to recover the ADC.
                {
                    bool wasOK = false;
                    double ADC = 0.0; //Will be around [0.88E-3 s/(mm*mm)] according to a paper I saw...

                    //----------------------------------------------- Linear Regression ----------------------------------------------
                    //This approach requires us to linearize the problem. This skews the uncertainties but lets us use
                    // an exact, fast, generic least-squares approach.
                    //
                    // To linearize, we assume voxel intensities satisfy:  S(i,j,k;b) = S(i,j,k;0) * exp(-b*ADC). 
                    // Taking a ln() of both sides, we end up with: ln(S) = ln(S_0) - b*ADC. 
                    // Thus using linear regression using {b,S} data, the slope will be [-ADC].
                    //
                    // The 'catch' is that the variances are transformed too, and the fitted model can be severely
                    // impacted. It is possible to transform the variances and use a weighted least-squares approach,
                    // but this approach can often lead to numerical difficulties due to the domain of the log. Both
                    // approaches can be tested by selecting an approach below.
                    //
                    samples_1D<double> linearized(channel_bval_course);
                    linearized.uncertainties_known_to_be_independent_and_random = channel_bval_course.uncertainties_known_to_be_independent_and_random;

                    bool Use_Weighted_LS = false; //Attempt to transform the uncertainties and use proper weighted least-squares. Often generates errors.
                    if(Use_Weighted_LS){
                        bool CanBeLinearized = true;
                        for(auto &datum : linearized.samples){
                            const auto S    = datum[2];
                            const auto dS   = datum[3];
                            const auto lnS  = std::log(S);
                            const auto dlnS = std::abs(dS/S); //Regardless of normality assumption regarding uncertainties.
                            if(!std::isfinite(lnS) || !std::isfinite(dlnS)){
                                CanBeLinearized = false;
                                break;
                            }
                            datum[2] = lnS;
                            datum[3] = dlnS;
                        }
                        if(CanBeLinearized){
                            auto res = linearized.Weighted_Linear_Least_Squares_Regression(&wasOK);
                            if(wasOK) ADC = -res.slope;
                        }

                    }else{ //Otherwise, use unweighted least-squares.
                        bool CanBeLinearized = true;
                        for(auto &datum : linearized.samples){
                            const auto S    = datum[2];
                            const auto lnS  = std::log(S);
                            if(!std::isfinite(lnS)){
                                CanBeLinearized = false;
                                break;
                            }
                            datum[1] = 0.0;
                            datum[2] = lnS;
                            datum[3] = 0.0;
                        }
                        if(CanBeLinearized){
                            const bool Skip_Extras = true;
                            auto res = linearized.Linear_Least_Squares_Regression(&wasOK, Skip_Extras);
                            if(wasOK) ADC = -res.slope;
                        }
                    }
                    
                    //--------------------------------------------- Non-Linear Regression --------------------------------------------
                    //Instead of linearizing, we can fit directly. This approach can get stuck in local optima, is more computationally 
                    // costly, and is strongly dependent on the starting parameters.
                    //
                    // NOTE: The function we pass in is not the objective function, it is 'f' in 
                    //         min( (F_i - f(x_i; vars))^2 ) 
                    //       where the minimization happens by adjusting 'vars'.
                    //
                    //auto min_func = [=](const std::list<double> &X, const std::list<double> &Vars) -> double {
                    //    const auto S0     = Vars.front()/500.0;
                    //    const auto invADC = Vars.back();
                    //    const auto bval   = X.front();
                    //    const auto out    = S0 * std::exp( - bval / invADC );
                    //    return std::isfinite(out) ? out : static_cast<double>( std::numeric_limits<float>::max() );
                    //};
                    //std::list<std::list<double>> packed_data;
                    //for(const auto &datum : channel_bval_course.samples) packed_data.push_back( { datum[0], datum[2] } );
                    //auto res = Ygor_Fit_LSS(&wasOK, min_func, packed_data, {1000.0, 1136.0}, 2, false, 100.0, 3000, 1E-4);
                    //if(wasOK){
                    //    //const auto rawS0  = std::get<0>(res).front();
                    //    //const auto S0     = rawS0/500.0;
                    //    const auto invADC = std::get<0>(res).back();
                    //    ADC = std::pow(invADC,-1);
                    //}

                    //Update the pixel value with the ADC, or handle the case of failure.
                    if(!wasOK){
                        //YLOGWARN("Encountered issue performing least squares. Continuing with voxel set to zero");
                        working.reference(row, col, chan) = std::numeric_limits<float>::quiet_NaN();
                        continue;
                    }else if(ADC < 0.0){
//                        //YLOGWARN("Least-squares resulted in a negative ADC. Continuing with voxel set to zero");
//YLOGERR("Need to figure out what to put here. Zero? NaN?");
//                        working.reference(row, col, chan) = std::numeric_limits<float>::quiet_NaN();
                        //Proceed with negative ADC. It is clearly not a valid result and should somehow be dealt with
                        // in later analyses.
                        working.reference(row, col, chan) = ADC;
                        continue;
                    }else{
//                        //Because the ADC is a small number in these units, scale it (effectively altering the units).
//                        // We want at least three significant figures, and the ADC should be ~0.9E-3.
//YLOGERR("Need to figure out if the 1E6x scaling factor is needed here. It might be for precision arguments");
//                        const auto ADC_int = static_cast<float>(ADC*1.0E6);
                        working.reference(row, col, chan) = ADC;

                        minmax_pixel.Digest(ADC);
                    }
                }
            }//Loop over channels.
        } //Loop over cols
    } //Loop over rows

    //Swap the original image with the working image.
    *first_img_it = working;

    UpdateImageDescription( std::ref(*first_img_it), "ADC" );
    UpdateImageWindowCentreWidth( std::ref(*first_img_it), minmax_pixel );

    return true;
}


