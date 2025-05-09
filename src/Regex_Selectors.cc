//Regex_Selectors.cc - A part of DICOMautomaton 2018. Written by hal clark.

#include "Regex_Selectors.h"

#include <string>
#include <list>
#include <numeric>
#include <functional>
#include <regex>
#include <optional>
#include <utility>
#include <cstdint>

#include "YgorString.h"
#include "YgorMath.h"

#include "Structs.h"

// ------------------------------------- Templates -------------------------------------

// Whitelist image arrays or point clouds using a limited vocabulary of specifiers.
// 
// Note: Positional specifiers (e.g., "first") act on the current whitelist. 
//       Beware when chaining filters!
template <class L> // L is a list of list::iterators of shared_ptr<Image_Array or Point_Cloud>.
L
Whitelist_Core( L lops,
           const std::string& Specifier,
           Regex_Selector_Opts Opts ){

    // Detect when the object holds contours, because they use reference wrappers rather than pointers.
    constexpr bool is_cc = std::is_same< decltype(lops),
                                         decltype(All_CCs( Drover() )) >::value;

    // Multiple key-value specifications stringified together.
    // For example, "key1@value1;key2@value2".
    do{
        const auto regex_split = Compile_Regex("^.*;.*$");
        if(!std::regex_match(Specifier, regex_split)) break; // Not a multi-key@value statement.

        auto v_kvs = SplitStringToVector(Specifier, ';', 'd');
        if(v_kvs.size() <= 1) throw std::logic_error("Unable to separate multiple key@value specifiers");

        for(auto & keyvalue : v_kvs){
            lops = Whitelist(lops, keyvalue, Opts);
        }
        return lops;
    }while(false);

    // A keyword and a single key name.
    // For example, "keymissing@key".
    do{
        const auto regex_split = Compile_Regex("^keymissing@.*$");
        if(!std::regex_match(Specifier, regex_split)) break; // Not a keymissing@key statement.
        
        auto v_k_v = SplitStringToVector(Specifier, '@', 'd');
        if(v_k_v.size() <= 1) throw std::logic_error("Unable to separate keymissing@key specifier");
        if(v_k_v.size() != 2) break; // Not a keymissing@key statement (hint: maybe multiple @'s present?).

        // Emulate this feature using a bogus regex that will never match when the key is present, but treat NAs as if
        // they match. So the only thing that will match are objects lacking this key.
        auto Opts_l = Opts;
        Opts_l.nas = Regex_Selector_Opts::NAs::Include;
        const std::string key = v_k_v.back();
        const std::string val = "gKNcTv4s5WXEsweUKIUqsDb7M0GvDI0J3G4LinJSKVYcSLg6V3GEQW2wa";

        lops = Whitelist(lops, key, val, Opts_l);
        return lops;
    }while(false);

    // Inverted regex key-value specifications stringified together.
    // For example, "!key@value".
    do{
        const auto regex_split = Compile_Regex("^[!].*@.*$");
        if(!std::regex_match(Specifier, regex_split)) break; // Not a key@value statement.
        
        auto v_k_v = SplitStringToVector(Specifier, '@', 'd');
        if(v_k_v.size() <= 1) throw std::logic_error("Unable to separate !key@value specifier");
        if(v_k_v.size() != 2) break; // Not a key@value statement (hint: maybe multiple @'s present?).

        auto lops_after = Whitelist(lops, v_k_v.front().substr(1), v_k_v.back(), Opts);
        for(const auto &l : lops_after){
            if constexpr ( is_cc ){
                lops.remove_if( [&](const auto &cc_refw){
                                    return (&(cc_refw.get()) == &(l.get()));
                                } );
            }else{
                lops.remove( l );
            }
        }

        return lops;
    }while(false);

    // A single key-value specifications stringified together.
    // For example, "key@value".
    do{
        const auto regex_split = Compile_Regex("^.*@.*$");
        if(!std::regex_match(Specifier, regex_split)) break; // Not a key@value statement.
        
        auto v_k_v = SplitStringToVector(Specifier, '@', 'd');
        if(v_k_v.size() <= 1) throw std::logic_error("Unable to separate key@value specifier");
        if(v_k_v.size() != 2) break; // Not a key@value statement (hint: maybe multiple @'s present?).

        lops = Whitelist(lops, v_k_v.front(), v_k_v.back(), Opts);
        return lops;
    }while(false);

    // Single-word positional specifiers, i.e. "all", "none", "first", "last", or zero-based 
    // numerical specifiers, e.g., "#0" (front), "#1" (second), "#-0" (last), and "#-1" (second-from-last).
    do{
        const auto regex_none  = Compile_Regex("^non?e?$");
        const auto regex_all   = Compile_Regex("^al?l?$");
        const auto regex_1st   = Compile_Regex("^fir?s?t?$");
        const auto regex_2nd   = Compile_Regex("^se?c?o?n?d?$");
        const auto regex_3rd   = Compile_Regex("^th?i?r?d?$");
        const auto regex_last  = Compile_Regex("^la?s?t?$");
        const auto regex_pnum  = Compile_Regex("^[#][0-9]+$");
        const auto regex_nnum  = Compile_Regex("^[#]-[0-9]+$");
        const auto regex_numer = Compile_Regex("^num?e?r?o?u?s?$");
        const auto regex_few   = Compile_Regex("^fewest?$");
        const auto regex_moret = Compile_Regex("^mor?e?[-_]?t?h?[ae]?n?[-_]?[(][-]?[0-9]+[)]$");
        const auto regex_fewt  = Compile_Regex("^fewer[-_]?t?h?[ae]?n?[-_]?[(][-]?[0-9]+[)]$");

        const auto regex_i_none  = Compile_Regex("^[!]non?e?$"); // Inverted variants of the above.
        const auto regex_i_all   = Compile_Regex("^[!]al?l?$");
        const auto regex_i_1st   = Compile_Regex("^[!]fir?s?t?$");
        const auto regex_i_2nd   = Compile_Regex("^[!]se?c?o?n?d?$");
        const auto regex_i_3rd   = Compile_Regex("^[!]th?i?r?d?$");
        const auto regex_i_last  = Compile_Regex("^[!]la?s?t?$");
        const auto regex_i_pnum  = Compile_Regex("^[!][#][0-9]+$");
        const auto regex_i_nnum  = Compile_Regex("^[!][#]-[0-9]+$");
        const auto regex_i_numer = Compile_Regex("^[!]num?e?r?o?u?s?$");
        const auto regex_i_few   = Compile_Regex("^[!]fewest?$");
        const auto regex_i_moret = Compile_Regex("^[!]mor?e?[-_]?t?h?[ae]?n?[-_]?[(][-]?[0-9]+[)]$");
        const auto regex_i_fewt  = Compile_Regex("^[!]fewer[-_]?t?h?[ae]?n?[-_]?[(][-]?[0-9]+[)]$");
        
        if(std::regex_match(Specifier, regex_i_none)){
            return lops;
        }
        if(std::regex_match(Specifier, regex_none)){
            lops.clear();
            return lops;
        }

        if(std::regex_match(Specifier, regex_i_all)){
            lops.clear();
            return lops;
        }
        if(std::regex_match(Specifier, regex_all)){
            return lops;
        }

        if( std::regex_match(Specifier, regex_i_1st)
        ||  std::regex_match(Specifier, regex_i_2nd) 
        ||  std::regex_match(Specifier, regex_i_3rd)  ){
            size_t N = 0; // Target.
            if( std::regex_match(Specifier, regex_i_1st) ){ N = 1;
            }else if( std::regex_match(Specifier, regex_i_2nd) ){ N = 2;
            }else if( std::regex_match(Specifier, regex_i_3rd) ){ N = 3;
            }else{
                throw std::logic_error("Regex positional specifier not understood. Cannot continue.");
            }

            decltype(lops) out;
            size_t i = 1;
            for(const auto &l : lops) if(N != i++) out.emplace_back(l);
            return out;
        }

        if( std::regex_match(Specifier, regex_1st)
        ||  std::regex_match(Specifier, regex_2nd) 
        ||  std::regex_match(Specifier, regex_3rd)  ){
            size_t N = 0; // Target.
            if( std::regex_match(Specifier, regex_1st) ){ N = 1;
            }else if( std::regex_match(Specifier, regex_2nd) ){ N = 2;
            }else if( std::regex_match(Specifier, regex_3rd) ){ N = 3;
            }else{
                throw std::logic_error("Regex positional specifier not understood. Cannot continue.");
            }

            decltype(lops) out;
            size_t i = 1;
            for(const auto &l : lops) if(N == i++) out.emplace_back(l);
            return out;
        }

        if(std::regex_match(Specifier, regex_i_last)){
            if(!lops.empty()) lops.pop_back();
            return lops;
        }
        if(std::regex_match(Specifier, regex_last)){
            decltype(lops) out;
            if(!lops.empty()) out.emplace_back(lops.back());
            return out;
        }

        if(std::regex_match(Specifier, regex_i_pnum)){
            auto pnum_extractor = std::regex("^[!][#]([0-9]+)$", std::regex::icase |
                                                                 std::regex::optimize |
                                                                 std::regex::extended);
            auto N = std::stoul(GetFirstRegex(Specifier, pnum_extractor));

            if(N < lops.size()){
                auto l_it = std::next( lops.begin(), N );
                lops.erase( l_it );
            }
            return lops;
        }
        if(std::regex_match(Specifier, regex_pnum)){
            auto pnum_extractor = std::regex("^[#]([0-9]+)$", std::regex::icase |
                                                              std::regex::optimize |
                                                              std::regex::extended);
            auto N = std::stoul(GetFirstRegex(Specifier, pnum_extractor));

            decltype(lops) out;
            if(N < lops.size()){
                auto l_it = std::next( lops.begin(), N );
                out.emplace_back(*l_it);
            }
            return out;
        }

        if(std::regex_match(Specifier, regex_i_nnum)){
            auto nnum_extractor = std::regex("^[!][#]-([0-9]+)$", std::regex::icase |
                                                                  std::regex::optimize |
                                                                  std::regex::extended);
            auto N = std::stoul(GetFirstRegex(Specifier, nnum_extractor));

            if(N < lops.size()) return lops;

            // Note: this one is slightly harder than the rest because you cannot directly erase() a reverse iterator.
            decltype(lops) out;
            size_t i = lops.size();
            for(auto l_it = lops.begin(); l_it != lops.end(); ++l_it, --i){
                if(i == N) continue;
                out.emplace_back(*l_it);
            }
            return out;
        }
        if(std::regex_match(Specifier, regex_nnum)){
            auto nnum_extractor = std::regex("^[#]-([0-9]+)$", std::regex::icase |
                                                               std::regex::optimize |
                                                               std::regex::extended);
            auto N = std::stoul(GetFirstRegex(Specifier, nnum_extractor));

            decltype(lops) out;
            if(N < lops.size()){
                auto l_it = std::next( lops.rbegin(), N );
                out.emplace_back(*l_it);
            }
            return out;
        }

        const auto extract_count = []( const typename decltype(lops)::value_type &l ) -> size_t {
            constexpr bool is_cc = std::is_same< decltype(lops),
                                                 decltype(All_CCs( Drover() )) >::value;

            if constexpr (is_cc){
                // Do nothing.
            }else{
                if( (*l) == nullptr ){
                    throw std::runtime_error("Encountered invalid pointer");
                }
            }

            size_t count = 0UL;

            if constexpr (is_cc){
                count = l.get().contours.size();

            }else if constexpr (std::is_same< decltype(lops),
                                              std::list<std::list<std::shared_ptr<Image_Array>>::iterator> >::value){
                count = (*l)->imagecoll.images.size();

            }else if constexpr (std::is_same< decltype(lops),
                                              std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator> >::value){
                count = (*l)->pset.points.size();

            }else if constexpr (std::is_same< decltype(lops),
                                              std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator> >::value){
                // Not exactly sure what to do here, so let's go for total number of elements needed to specify
                // the mesh, which is approximately related to the the number of bytes needed for storage (i.e.,
                // one type of 'size').
                count = (*l)->meshes.vertices.size() + (*l)->meshes.faces.size();

            }else if constexpr (std::is_same< decltype(lops),
                                              std::list<std::list<std::shared_ptr<RTPlan>>::iterator> >::value){
                const auto count_static_keyframes = [](const RTPlan &t) -> size_t {
                                                        size_t c = 0;
                                                        for(const auto &ds : t.dynamic_states) c += ds.static_states.size();
                                                        return c;
                                                    };
                count = count_static_keyframes(*(*l));

            }else if constexpr (std::is_same< decltype(lops),
                                              std::list<std::list<std::shared_ptr<Line_Sample>>::iterator> >::value){
                count = (*l)->line.samples.size();

            }else{
                throw std::invalid_argument("The 'more-than' and 'fewer-than' selectors are not implemented for this data type");
            }
            return count;
        };

        // 'Numerous' and 'fewest' selectors.
        {
            const bool numerous_pos = std::regex_match(Specifier, regex_numer);
            const bool numerous_neg = std::regex_match(Specifier, regex_i_numer);

            const bool fewest_pos = std::regex_match(Specifier, regex_few);
            const bool fewest_neg = std::regex_match(Specifier, regex_i_few);

            if( numerous_pos || numerous_neg || fewest_pos || fewest_neg ){

                if(lops.empty()) return lops;

                auto m = std::max_element( std::begin(lops), std::end(lops),
                                           [=]( const typename decltype(lops)::value_type &l,
                                                const typename decltype(lops)::value_type &r ) -> bool {
                    if constexpr (is_cc){
                        // Do nothing.
                    }else{
                        if( ( (*l) == nullptr )
                        ||  ( (*r) == nullptr ) ){
                            throw std::runtime_error("Encountered invalid pointer");
                        }
                    }

                    const auto sort_order = [&](size_t l, size_t r) -> bool {
                        return (numerous_pos || numerous_neg) ? (l < r) : (r < l);
                    };

                    const auto N_l = extract_count(l);
                    const auto N_r = extract_count(r);

                    return sort_order(N_l, N_r);
                } );

                decltype(lops) largest;
                largest.splice( std::end(largest), lops, m );

                return (numerous_pos || fewest_pos) ? largest : lops;
            }
        }

        // 'more_than(N)', 'fewer_than(N)', and inverted selectors.
        {
            const bool selector_moret = std::regex_match(Specifier, regex_moret);
            const bool selector_fewt  = std::regex_match(Specifier, regex_fewt);

            const bool selector_i_moret = std::regex_match(Specifier, regex_i_moret);
            const bool selector_i_fewt  = std::regex_match(Specifier, regex_i_fewt);

            if( selector_moret || selector_fewt || selector_i_moret || selector_i_fewt ){
                if(lops.empty()) return lops;

                const auto num_extractor = std::regex(".*[(]([-]?[0-9]+)[)]$", std::regex::icase |
                                                                               std::regex::optimize |
                                                                               std::regex::extended);
                const auto N = std::stol(GetFirstRegex(Specifier, num_extractor));

                const auto eval = [&](size_t count) -> bool {
                    const bool is_moret = (N < static_cast<int64_t>(count));
                    const bool is_fewt  = (static_cast<int64_t>(count) < N);
                    return (selector_moret) ? is_moret :
                           (selector_fewt)  ? is_fewt :
                           (selector_i_moret) ? !is_moret :
                           (selector_i_fewt)  ? !is_fewt : false;
                };

                decltype(lops) out;
                for(const auto &l : lops){
                    if constexpr (is_cc){
                        // Do nothing.
                    }else{
                        if( (*l) == nullptr ){
                            throw std::runtime_error("Encountered invalid pointer");
                        }
                    }
                    const size_t count = extract_count(l);

                    if(eval(count)){
                        out.emplace_back(l);
                    }
                }
                return out;
            }
        }

    }while(false);

    throw std::invalid_argument("Selection is not valid. Cannot continue.");
    decltype(lops) out;
    return out;
}

// This is a convenience routine to combine multiple filtering passes into a single logical statement.
template <class L> // L is a list of list::iterators of shared_ptr<Image_Array or Point_Cloud>.
L
Whitelist_Core( L lops,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    for(const auto& kv_pair : MetadataKeyValueRegex){
        lops = Whitelist(lops, kv_pair.first, kv_pair.second, Opts);
    }
    return lops;
}


// --------------------------------------- Misc. ---------------------------------------

// Compile and return a regex using the application-wide default settings.
std::regex
Compile_Regex(const std::string& input){
    return std::regex(input, std::regex::icase | 
                             std::regex::nosubs |
                             std::regex::optimize |
#ifdef DCMA_CPPSTDLIB_HAS_REGEX_MULTILINE
                             // This symbol may be absent for older toolchains.
                             // See https://gcc.gnu.org/pipermail/libstdc++/2021-September/053209.html for libstdc++.
                             std::regex_constants::multiline |
#endif
                             std::regex::ECMAScript);
}

// A class for managing multiple mutually-exclusive regexes, e.g., method selectors.
regex_group::regex_group() : prefix_length(2) {};

std::string regex_group::insert(std::string n){

    // Splits the key into parts along hyphens and spaces.
    const auto split = [](std::string x){
        auto s = SplitStringToVector(x, '-', 'd');
        s = SplitVector(s, '_', 'd');
        s = SplitVector(s, ' ', 'd');
        return s;
    };

    // Trims the prefix vector.
    const auto trim = [](std::vector<std::string> x,
                         int64_t l){
        for(auto &s : x){
            s = Get_Preceeding_Chars(s, l);
        }
        return x;
    };

    // Applies in-band characters to control the regex match.
    const auto decorate = [&split,
                           &trim ](std::string x,
                                   int64_t l){
        std::string out;
        out += "^";

        auto strs = split(x);
        auto prfx = trim(strs, l);

        for(size_t i = 0UL; i < strs.size(); ++i){
            // Add separator.
            if(i != 0UL){
                out += "[-_ ]?";
            }

            // Add the prefix characters, which are mandatory in the regex pattern.
            out += prfx.at(i);

            // Add the remaining characters, which are optional in the regex.
            for(size_t j = prfx.at(i).size(); j < strs.at(i).size(); ++j){
                out += strs[i].at(j);
                out += "?";
            }
        }

        out += "$";
        return std::make_pair(out, prfx);
    };

    // Starting with only the current pattern in the bag, attempt to generate a decoration
    // and prefix that is distinct from all prior patterns.
    std::set<std::string> bag;

    // Confirm that the input is not already an exact duplicates.
    bool is_duplicate = false;
    for(const auto &p : this->regexes){
        if(p.first == n){
            is_duplicate = true;
            break;
        }
    }
    if(is_duplicate){
        YLOGDEBUG("Input '" << n << "' is a duplicate, skipping it");
    }else{
        bag.insert(n);
    }

    while(!bag.empty()){
        std::map<std::string, std::string> l_decorations;
        std::map<std::vector<std::string>, std::string> l_prefixes;

        for(const auto &pattern : bag){
            const auto [decorated, prefixes] = decorate(pattern, this->prefix_length);
            l_decorations[pattern] = decorated;
            l_prefixes[prefixes] = pattern;
            YLOGDEBUG("Generated regex '" << decorated << "' from pattern '" << pattern << "'");
        }

        const bool unique_in_bag = (l_prefixes.size() == bag.size());
        this->prefixes.merge(l_prefixes);
        const bool unique_globally = l_prefixes.empty();

        // Check if there are any conflicts within the bag or with all existing patterns.
        if( unique_in_bag 
        &&  unique_globally ){
            // No conflict detected. Insert the decorations and terminate the loop.
            for(const auto &d : l_decorations){
                this->regexes.insert({d.first, Compile_Regex(d.second)});
            }
            bag.clear();
            break;

        // Otherwise, increase the prefix length, enroll *all* patterns in the bag, and loop.
        }else{
            ++this->prefix_length;
            YLOGDEBUG("Detected conflict with input '" << n
                      << "', increasing prefix length to " << this->prefix_length
                      << " and trying again");

            for(const auto &p : this->regexes){
                bag.insert(p.first);
            }
            this->regexes.clear();
            this->prefixes.clear();
        }

        if(10 < this->prefix_length){
            throw std::runtime_error("Unable to orthogonalize inputs");
        }
    }

    return n;
}

const std::regex & regex_group::locate(const std::string &n) const {
    return this->regexes.at(n);
}

bool regex_group::matches(const std::string &raw, const std::string &known) const {
    return std::regex_match(raw, this->regexes.at(known));
}



// Human-readable information about how selectors can be specified.
static
std::string
GenericSelectionInfo(const std::string &name_of_unit,
                     const std::string &name_of_subobject){
    return " Selection specifiers can be of three types: positional, metadata-based key@value regex, and intrinsic."_s
         + "\n\n"_s 
         + "Positional specifiers can be 'first', 'last', 'none', or 'all' literals."_s
         + " Additionally '#N' for some positive integer N selects the Nth "_s + name_of_unit 
         + " (with zero-based indexing)."_s
         + " Likewise, '#-N' selects the Nth-from-last "_s + name_of_unit + "."_s
         + " Positional specifiers can be inverted by prefixing with a '!'."_s
         + "\n\n"_s 
         + "Metadata-based key@value expressions are applied by matching the keys verbatim and the values with regex."_s
         + " In order to invert metadata-based selectors, the regex logic must be inverted"_s
         + " (i.e., you can *not* prefix metadata-based selectors with a '!')."_s
         + " Note regexes are case insensitive and should use extended POSIX syntax."_s
         + "\n\n"_s 
         + "Intrinsic specifiers can be 'numerous', 'fewest', 'more-than(N)', and 'fewer-than(N)'."_s
         + " Literals 'numerous' and 'fewest' select the "_s + name_of_unit 
         + " composed of the greatest and fewest number of "_s + name_of_subobject + "."_s
         + " Only one or zero "_s + name_of_unit + " will be selected;"_s
         + " if there are ties, there is no guarantee which "_s + name_of_unit + " will be selected."_s
         + " Use 'more-than(N)' or 'fewer-than(N)' to select multiple "_s + name_of_unit
         + " based on a threshold count, i.e., where all selected "_s + name_of_unit
         + " have more than or fewer than $N$ "_s + name_of_subobject + "."_s
         + " Intrinsic specifiers can be inverted by prefixing with a '!'."_s
         + " Note that '!numerous' means all "_s + name_of_unit
         + " that do not have the greatest number of "_s + name_of_subobject + ","_s
         + " not the least-numerous "_s + name_of_unit + " (i.e., 'fewest')."_s
         + "\n\n"_s 
         + "All criteria (positional, metadata, and intrinsic) can be mixed together."_s
         + " Multiple criteria can be specified by separating them with a ';' and are applied in the order specified."_s;
}

// ---------------------------------- Contours / ROIs ----------------------------------

// Stuff references to all contour collections into a list.
//
// Note: the output is meant to be filtered out using the selectors below.
std::list<std::reference_wrapper<contour_collection<double>>>
All_CCs( Drover DICOM_data ){
    std::list<std::reference_wrapper<contour_collection<double>>> cc_all;
    if(DICOM_data.contour_data != nullptr){
        for(auto & cc : DICOM_data.contour_data->ccs){
            if(cc.contours.empty()) continue;
            auto base_ptr = reinterpret_cast<contour_collection<double> *>(&cc);
            if(base_ptr == nullptr) continue;
            cc_all.push_back( std::ref(*base_ptr) );
        }
    }
    return cc_all;
}


// Whitelist contour collections using the provided regex.
std::list<std::reference_wrapper<contour_collection<double>>>
Whitelist( std::list<std::reference_wrapper<contour_collection<double>>> ccs,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    ccs.remove_if([&](std::reference_wrapper<contour_collection<double>> cc) -> bool {
        if(cc.get().contours.empty()) return true; // Remove collections containing no contours.

        if(Opts.validation == Regex_Selector_Opts::Validation::Representative){
            auto ValueOpt = cc.get().contours.front().GetMetadataValueAs<std::string>(MetadataKey);
            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("Regex selector representative->NAs option not understood. Cannot continue.");

        }else if(Opts.validation == Regex_Selector_Opts::Validation::Pedantic){
            const auto Values = cc.get().get_distinct_values_for_key(MetadataKey);

            if(Values.empty()){
                if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                    return false;
                }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                    return true;
                }
                throw std::logic_error("Regex selector pedantic->NAs option not understood. Cannot continue.");

            }else{
                for(const auto & Value : Values){
                    if( !std::regex_match(Value,theregex) ) return true;
                }
                return false;
            }
            throw std::logic_error("Regex selector pedantic option not understood. Cannot continue.");

        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return ccs;
}

// This is a convenience routine to combine multiple filtering passes into a single logical statement.
std::list<std::reference_wrapper<contour_collection<double>>>
Whitelist( std::list<std::reference_wrapper<contour_collection<double>>> ccs,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    for(const auto& kv_pair : MetadataKeyValueRegex){
        ccs = Whitelist(ccs, kv_pair.first, kv_pair.second, Opts);
    }
    return ccs;
}

// Whitelist contour ROIs using a limited vocabulary of specifiers.
std::list<std::reference_wrapper<contour_collection<double>>>
Whitelist( std::list<std::reference_wrapper<contour_collection<double>>> ccs,
           std::string Specifier,
           Regex_Selector_Opts Opts){
    return Whitelist_Core( std::move(ccs), std::move(Specifier), Opts );
}


// Utility function that accepts both optionally-valid regex selectors and an optionally-valid metadata selector.
std::list<std::reference_wrapper<contour_collection<double>>>
Whitelist( std::list<std::reference_wrapper<contour_collection<double>>> ccs,
           std::optional<std::string> ROILabelRegexOpt,
           std::optional<std::string> NormalizedROILabelRegexOpt,
           std::optional<std::string> SpecifierOpt,
           Regex_Selector_Opts Opts){

    if(SpecifierOpt){
        ccs = Whitelist(ccs, SpecifierOpt.value(), Opts);
    }
    if(ROILabelRegexOpt){
        ccs = Whitelist(ccs, "ROIName",
                             ROILabelRegexOpt.value(), Opts);
    }
    if(NormalizedROILabelRegexOpt){
        ccs = Whitelist(ccs, "NormalizedROIName",
                             NormalizedROILabelRegexOpt.value(), Opts);
    }

    if( !SpecifierOpt
    &&  !ROILabelRegexOpt
    &&  !NormalizedROILabelRegexOpt ){
        ccs.clear();
    }
    return ccs;
}

// Utility functions documenting the contour whitelist routines for operations.
OperationArgDoc RCWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "ROILabelRegex";
    out.desc = "A regular expression (regex) matching *raw* ROI contour labels/names to consider."
               "\n\n"
               "Selection is performed on a whole-ROI basis; individual contours cannot be selected."
               " Be aware that input spaces are trimmed to a single space."
               " If your ROI name has more than two sequential spaces, use regular expressions or escaping to avoid them."
               " All ROIs you want to select must match the provided (single) regex, so use boolean or ('|') if needed."
               "\n\n"
               " The regular expression engine is case insensitive and uses a C++ modified ECMAScript grammar which"
               " is documented at <https://en.cppreference.com/w/cpp/regex/ecmascript>."
               " Note that '.*' will match all available ROIs and '^(?!xyz).*$' will match all except 'xyz'."
               "\n\n"
               "Note that this parameter will match 'raw' contour labels.";
    out.examples = { ".*", ".*body.*", "body", "^body$", "Liver",
                     R"***(.*left.*parotid.*|.*right.*parotid.*|.*eyes.*)***",
                     R"***(left_parotid|right_parotid)***",
                     R"***(^(?!left_parotid).*$)***"};
    out.default_val = ".*";
    out.expected = true;

    return out;
}

OperationArgDoc NCWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "NormalizedROILabelRegex";
    out.desc = "A regular expression (regex) matching *normalized* ROI contour labels/names to consider."
               "\n\n"
               "Selection is performed on a whole-ROI basis; individual contours cannot be selected."
               " Be aware that input spaces are trimmed to a single space."
               " If your ROI name has more than two sequential spaces, use regular expressions or escaping to avoid them."
               " All ROIs you want to select must match the provided (single) regex, so use boolean or ('|') if needed."
               "\n\n"
               " The regular expression engine is case insensitive and uses a C++ modified ECMAScript grammar which"
               " is documented at <https://en.cppreference.com/w/cpp/regex/ecmascript>."
               " Note that '.*' will match all available ROIs and '^(?!xyz).*$' will match all except 'xyz'."
               "\n\n"
               "Note that this parameter will match contour labels that have been"
               " *normalized* (i.e., mapped, translated) using the user-provided provided lexicon."
               " This is useful for handling data with heterogeneous naming conventions where fuzzy matching is required."
               " Refer to the lexicon for available labels.";
    out.examples = { ".*", ".*Body.*", "Body", "liver",
                     R"***(.*Left.*Parotid.*|.*Right.*Parotid.*|.*Eye.*)***",
                     R"***(Left Parotid|Right Parotid)***",
                     R"***(^(?!Left Parotid).*$)***"};
    out.default_val = ".*";
    out.expected = true;

    return out;
}

OperationArgDoc CCWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "ROISelection";
    out.desc = "Select one or more contour regions of interest (aka contour collection)."_s
               + " Note that each region of interest may be comprised of multiple individual contours."_s
               + GenericSelectionInfo("contour collections", "contours");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first",
                     "numerous", "fewest", "more-than(5)", "!fewer-than(10)" };

    return out;
}

// ----------------------------------- Image Arrays ------------------------------------

// Provide iterators for all image arrays into a list.
//
// Note: The output is meant to be filtered out using the selectors below.
//
// Note: List iterators are provided because it is common to need to shuffle image ordering around.
//       The need appears to be less common for contours, so the interface is slightly different
//       compared to the contour whitelist interface.
std::list<std::list<std::shared_ptr<Image_Array>>::iterator>
All_IAs( Drover &DICOM_data ){
    std::list<std::list<std::shared_ptr<Image_Array>>::iterator> ia_all;

    for(auto iap_it = DICOM_data.image_data.begin(); iap_it != DICOM_data.image_data.end(); ++iap_it){
        if((*iap_it) == nullptr) continue;
        ia_all.push_back(iap_it);
    }
    return ia_all;
}

// Whitelist image arrays using the provided regex.
std::list<std::list<std::shared_ptr<Image_Array>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Image_Array>>::iterator> ias,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    ias.remove_if([&](std::list<std::shared_ptr<Image_Array>>::iterator iap_it) -> bool {
        if((*iap_it) == nullptr) return true;
        if((*iap_it)->imagecoll.images.empty()) return true; // Remove arrays containing no images.

        if(Opts.validation == Regex_Selector_Opts::Validation::Representative){
            auto ValueOpt = (*iap_it)->imagecoll.images.front().GetMetadataValueAs<std::string>(MetadataKey);
            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("Regex selector representative->NAs option not understood. Cannot continue.");

        }else if(Opts.validation == Regex_Selector_Opts::Validation::Pedantic){
            const auto Values = (*iap_it)->imagecoll.get_distinct_values_for_key(MetadataKey);

            if(Values.empty()){
                if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                    return false;
                }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                    return true;
                }
                throw std::logic_error("Regex selector pedantic->NAs option not understood. Cannot continue.");

            }else{
                for(const auto & Value : Values){
                    if( !std::regex_match(Value,theregex) ) return true;
                }
                return false;
            }
            throw std::logic_error("Regex selector pedantic option not understood. Cannot continue.");

        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return ias;
}

// Whitelist image arrays using a limited vocabulary of specifiers.
std::list<std::list<std::shared_ptr<Image_Array>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Image_Array>>::iterator> ias,
           std::string Specifier,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(ias), std::move(Specifier), Opts );
}


// This is a convenience routine to combine multiple filtering passes into a single logical statement.
std::list<std::list<std::shared_ptr<Image_Array>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Image_Array>>::iterator> ias,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(ias), MetadataKeyValueRegex, Opts );
}


// Utility function documenting the image array whitelist routines for operations.
OperationArgDoc IAWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "ImageSelection";
    out.desc = "Select one or more image arrays."_s
               + " Note that image arrays can hold anything, but will typically represent a single contiguous"_s
               + " 3D volume (i.e., a volumetric CT scan) or '4D' time-series."_s
               + " Be aware that it is possible to mix logically unrelated images together."_s
               + GenericSelectionInfo("image array", "images");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first",
                     "numerous", "fewest", "more-than(5)", "!fewer-than(10)" };

    return out;
}

// ----------------------------------- Point Clouds ------------------------------------

// Provide pointers for all point clouds into a list.
//
// Note: The output is meant to be filtered using the selectors below.
std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator>
All_PCs( Drover &DICOM_data ){
    std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator> pc_all;

    for(auto pcp_it = DICOM_data.point_data.begin(); pcp_it != DICOM_data.point_data.end(); ++pcp_it){
        if((*pcp_it) == nullptr) continue;
        pc_all.push_back(pcp_it);
    }
    return pc_all;
}


// Whitelist point clouds using the provided regex.
std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator> pcs,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    pcs.remove_if([&](std::list<std::shared_ptr<Point_Cloud>>::iterator pcp_it) -> bool {
        if((*pcp_it) == nullptr) return true;
        if((*pcp_it)->pset.points.empty()) return true; // Remove arrays containing no images.

        if( // Note: Point_Clouds are dissimilar to Image_Arrays in that individual images can have different
                  //       metadata, but point clouds cannot. We keep these options for consistency.
                  (Opts.validation == Regex_Selector_Opts::Validation::Representative)
              ||  (Opts.validation == Regex_Selector_Opts::Validation::Pedantic)        ){

            auto ValueOpt = (*pcp_it)->pset.GetMetadataValueAs<std::string>(MetadataKey);
            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("NAs option not understood. Cannot continue.");
        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return pcs;
}


// Whitelist point clouds using a limited vocabulary of specifiers.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator> pcs,
           std::string Specifier,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(pcs), std::move(Specifier), Opts );
}    


// This is a convenience routine to combine multiple filtering passes into a single logical statement.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Point_Cloud>>::iterator> pcs,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(pcs), MetadataKeyValueRegex, Opts );
}


// Utility function documenting the point cloud whitelist routines for operations.
OperationArgDoc PCWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "PointSelection";
    out.desc = "Select one or more point clouds."_s
               + " Note that point clouds can hold a variety of data with varying attributes,"_s
               + " but each point cloud is meant to represent a single logically cohesive collection of points."_s
               + " Be aware that it is possible to mix logically unrelated points together."_s
               + GenericSelectionInfo("point cloud", "vertices");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first",
                     "numerous", "fewest", "more-than(5)", "!fewer-than(10)" };

    return out;
}

// ----------------------------------- Surface Meshes ------------------------------------

// Provide pointers for all surface meshes into a list.
//
// Note: The output is meant to be filtered using the selectors below.
std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator>
All_SMs( Drover &DICOM_data ){
    std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator> sm_all;

    for(auto smp_it = DICOM_data.smesh_data.begin(); smp_it != DICOM_data.smesh_data.end(); ++smp_it){
        if((*smp_it) == nullptr) continue;
        sm_all.push_back(smp_it);
    }
    return sm_all;
}


// Whitelist surface meshes using the provided regex.
std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator> sms,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    sms.remove_if([&](std::list<std::shared_ptr<Surface_Mesh>>::iterator smp_it) -> bool {
        if((*smp_it) == nullptr) return true;
        if((*smp_it)->meshes.vertices.empty()) return true; // Remove meshes containing no vertices.
        if((*smp_it)->meshes.faces.empty()) return true; // Remove meshes containing no faces.

        if( // Note: A single Surface_Mesh corresponds to one individual metadata store. While a single
                  //       Surface_Mesh can be comprised of multiple disconnected meshes, they are herein considered
                  //       to be part of the same logical group. As for Point_Clouds, we keep the following options for
                  //       consistency.
                  (Opts.validation == Regex_Selector_Opts::Validation::Representative)
              ||  (Opts.validation == Regex_Selector_Opts::Validation::Pedantic)        ){

            std::optional<std::string> ValueOpt 
                    = ( (*smp_it)->meshes.metadata.count(MetadataKey) != 0 ) ?
                      (*smp_it)->meshes.metadata[MetadataKey] :
                      std::optional<std::string>();
            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("NAs option not understood. Cannot continue.");
        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return sms;
}


// Whitelist surface meshes using a limited vocabulary of specifiers.
//
// Note: this routine shares the generic Image_Arrays and Point_Clouds implementation above.
std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator> sms,
           std::string Specifier,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(sms), std::move(Specifier), Opts );
}    


// This is a convenience routine to combine multiple filtering passes into a single logical statement.
//
// Note: this routine shares the generic Image_Arrays and Point_Clouds implementation above.
std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Surface_Mesh>>::iterator> sms,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(sms), MetadataKeyValueRegex, Opts );
}


// Utility function documenting the surface mesh whitelist routines for operations.
OperationArgDoc SMWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "MeshSelection";
    out.desc = "Select one or more surface meshes."_s
               + " Note that a single surface mesh may hold many disconnected mesh components;"_s
               + " they should collectively represent a single logically cohesive object."_s
               + " Be aware that it is possible to mix logically unrelated sub-meshes together in a single mesh."_s
               + GenericSelectionInfo("surface mesh", "elements (vertices + faces)");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first",
                     "numerous", "fewest", "more-than(5)", "!fewer-than(10)" };

    return out;
}

// ------------------------------------ RTPlan -------------------------------------

// Provide pointers for all treatment plans into a list.
//
// Note: The output is meant to be filtered using the selectors below.
std::list<std::list<std::shared_ptr<RTPlan>>::iterator>
All_TPs( Drover &DICOM_data ){
    std::list<std::list<std::shared_ptr<RTPlan>>::iterator> tp_all;

    for(auto tpp_it = DICOM_data.rtplan_data.begin(); tpp_it != DICOM_data.rtplan_data.end(); ++tpp_it){
        if((*tpp_it) == nullptr) continue;
        tp_all.push_back(tpp_it);
    }
    return tp_all;
}


// Whitelist treatment plans using the provided regex.
std::list<std::list<std::shared_ptr<RTPlan>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<RTPlan>>::iterator> tps,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    tps.remove_if([&](std::list<std::shared_ptr<RTPlan>>::iterator tpp_it) -> bool {
        if((*tpp_it) == nullptr) return true;
        if((*tpp_it)->dynamic_states.empty()) return true; // Remove plans containing no beams.

        if( // Note: A RTPlan corresponds to one individual metadata store. While a single
                  //       RTPlan can be comprised of multiple disconnected beams, they are 
                  //       herein considered to be part of the same logical group.
                  (Opts.validation == Regex_Selector_Opts::Validation::Representative)
              ||  (Opts.validation == Regex_Selector_Opts::Validation::Pedantic)        ){

            std::optional<std::string> ValueOpt 
                    = ( (*tpp_it)->metadata.count(MetadataKey) != 0 ) ?
                      (*tpp_it)->metadata[MetadataKey] :
                      std::optional<std::string>();

            // TODO: support selection of Dynamic_Machine_State and Static_Machine_State metadata too.

            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("NAs option not understood. Cannot continue.");
        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return tps;
}


// Whitelist treatment plans using a limited vocabulary of specifiers.
//
// Note: this routine shares the generic Image_Arrays, Point_Clouds, and Surface_Mesh implementation above.
std::list<std::list<std::shared_ptr<RTPlan>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<RTPlan>>::iterator> tps,
           std::string Specifier,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(tps), std::move(Specifier), Opts );
}    


// This is a convenience routine to combine multiple filtering passes into a single logical statement.
//
// Note: this routine shares the generic Image_Arrays, Point_Clouds, and Surface_Mesh implementation above.
std::list<std::list<std::shared_ptr<RTPlan>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<RTPlan>>::iterator> tps,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(tps), MetadataKeyValueRegex, Opts );
}


// Utility function documenting the treatment plan whitelist routines for operations.
OperationArgDoc TPWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "RTPlanSelection";
    out.desc = "Select one or more treatment plans."_s
               + " Note that a single treatment plan may be composed of multiple beams;"_s
               + " if delivered sequentially, they should collectively represent a single logically cohesive plan."_s
               + GenericSelectionInfo("treatment plan", "control points");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first",
                     "numerous", "fewest", "more-than(5)", "!fewer-than(10)" };

    return out;
}

// ----------------------------------- Line Samples ------------------------------------

// Provide pointers for all line samples into a list.
//
// Note: The output is meant to be filtered using the selectors below.
std::list<std::list<std::shared_ptr<Line_Sample>>::iterator>
All_LSs( Drover &DICOM_data ){
    std::list<std::list<std::shared_ptr<Line_Sample>>::iterator> ls_all;

    for(auto lsp_it = DICOM_data.lsamp_data.begin(); lsp_it != DICOM_data.lsamp_data.end(); ++lsp_it){
        if((*lsp_it) == nullptr) continue;
        ls_all.push_back(lsp_it);
    }
    return ls_all;
}


// Whitelist line samples using the provided regex.
std::list<std::list<std::shared_ptr<Line_Sample>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Line_Sample>>::iterator> lss,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    lss.remove_if([&](std::list<std::shared_ptr<Line_Sample>>::iterator lsp_it) -> bool {
        if((*lsp_it) == nullptr) return true;
        if((*lsp_it)->line.samples.empty()) return true; // Remove arrays containing no samples.

        if( // Note: Line_Samples are dissimilar to Image_Arrays in that individual images can have different
                  //       metadata, but point clouds cannot. We keep these options for consistency.
                  (Opts.validation == Regex_Selector_Opts::Validation::Representative)
              ||  (Opts.validation == Regex_Selector_Opts::Validation::Pedantic)        ){

            std::optional<std::string> ValueOpt 
                    = ( (*lsp_it)->line.metadata.count(MetadataKey) != 0 ) ?
                      (*lsp_it)->line.metadata[MetadataKey] :
                      std::optional<std::string>();
            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("NAs option not understood. Cannot continue.");
        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return lss;
}


// Whitelist line samples using a limited vocabulary of specifiers.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Line_Sample>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Line_Sample>>::iterator> lss,
           std::string Specifier,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(lss), std::move(Specifier), Opts );
}    


// This is a convenience routine to combine multiple filtering passes into a single logical statement.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Line_Sample>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Line_Sample>>::iterator> lss,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(lss), MetadataKeyValueRegex, Opts );
}


// Utility function documenting the point cloud whitelist routines for operations.
OperationArgDoc LSWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "LSampSelection";
    out.desc = "Select one or more line samples."_s
               + GenericSelectionInfo("line sample", "samples");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first",
                     "numerous", "fewest", "more-than(5)", "!fewer-than(10)" };

    return out;
}

// ------------------------------------ Transform3 -------------------------------------

// Provide pointers for all transforms into a list.
//
// Note: The output is meant to be filtered using the selectors below.
std::list<std::list<std::shared_ptr<Transform3>>::iterator>
All_T3s( Drover &DICOM_data ){
    std::list<std::list<std::shared_ptr<Transform3>>::iterator> t3_all;

    for(auto t3p_it = DICOM_data.trans_data.begin(); t3p_it != DICOM_data.trans_data.end(); ++t3p_it){
        if((*t3p_it) == nullptr) continue;
        t3_all.push_back(t3p_it);
    }
    return t3_all;
}


// Whitelist transforms using the provided regex.
std::list<std::list<std::shared_ptr<Transform3>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Transform3>>::iterator> t3s,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    t3s.remove_if([&](std::list<std::shared_ptr<Transform3>>::iterator t3p_it) -> bool {
        if((*t3p_it) == nullptr) return true;
        if(std::holds_alternative<std::monostate>( (*t3p_it)->transform )) return true; // Remove empty transforms.

        if( (Opts.validation == Regex_Selector_Opts::Validation::Representative)
              ||  (Opts.validation == Regex_Selector_Opts::Validation::Pedantic)        ){

            std::optional<std::string> ValueOpt 
                    = ( (*t3p_it)->metadata.count(MetadataKey) != 0 ) ?
                      (*t3p_it)->metadata[MetadataKey] :
                      std::optional<std::string>();
            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("NAs option not understood. Cannot continue.");
        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return t3s;
}


// Whitelist transforms using a limited vocabulary of specifiers.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Transform3>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Transform3>>::iterator> t3s,
           std::string Specifier,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(t3s), std::move(Specifier), Opts );
}    


// This is a convenience routine to combine multiple filtering passes into a single logical statement.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Transform3>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Transform3>>::iterator> t3s,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(t3s), MetadataKeyValueRegex, Opts );
}


// Utility function documenting the transform whitelist routines for operations.
OperationArgDoc T3WhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "TransformSelection";
    out.desc = "Select one or more transform objects (aka 'warp' objects)."_s
               + GenericSelectionInfo("transformation", "sub-objects");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first" };

    return out;
}

// ----------------------------------- Sparse Tables ------------------------------------

// Provide pointers for all line samples into a list.
//
// Note: The output is meant to be filtered using the selectors below.
std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator>
All_STs( Drover &DICOM_data ){
    std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator> st_all;

    for(auto stp_it = DICOM_data.table_data.begin(); stp_it != DICOM_data.table_data.end(); ++stp_it){
        if((*stp_it) == nullptr) continue;
        st_all.push_back(stp_it);
    }
    return st_all;
}


// Whitelist tables using the provided regex.
std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator> sts,
           std::string MetadataKey,
           std::string MetadataValueRegex,
           Regex_Selector_Opts Opts ){

    auto theregex = Compile_Regex(std::move(MetadataValueRegex));

    sts.remove_if([&](std::list<std::shared_ptr<Sparse_Table>>::iterator stp_it) -> bool {
        if((*stp_it) == nullptr) return true;
        //if((*stp_it)->table.data().empty()) return true; // Remove tables containing no cells.

        if( // Note: Sparse_Tables are dissimilar to Image_Arrays in that individual images can have different
            //       metadata, but tables are 1-to-1. We keep these options for consistency.
                  (Opts.validation == Regex_Selector_Opts::Validation::Representative)
              ||  (Opts.validation == Regex_Selector_Opts::Validation::Pedantic)        ){

            std::optional<std::string> ValueOpt 
                    = ( (*stp_it)->table.metadata.count(MetadataKey) != 0 ) ?
                      (*stp_it)->table.metadata[MetadataKey] :
                      std::optional<std::string>();
            if(ValueOpt){
                return !(std::regex_match(ValueOpt.value(),theregex));
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Include){
                return false;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::Exclude){
                return true;
            }else if(Opts.nas == Regex_Selector_Opts::NAs::TreatAsEmpty){
                return !(std::regex_match("",theregex));
            }
            throw std::logic_error("NAs option not understood. Cannot continue.");
        }
        throw std::logic_error("Regex selector option not understood. Cannot continue.");
        return true; // Should never get here.
    });
    return sts;
}


// Whitelist sparse tables using a limited vocabulary of specifiers.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator> sts,
           std::string Specifier,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(sts), std::move(Specifier), Opts );
}    


// This is a convenience routine to combine multiple filtering passes into a single logical statement.
//
// Note: this routine shares the generic Image_Arrays implementation above.
std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator>
Whitelist( std::list<std::list<std::shared_ptr<Sparse_Table>>::iterator> sts,
           std::list< std::pair<std::string, 
                                std::string> > MetadataKeyValueRegex,
           Regex_Selector_Opts Opts ){

    return Whitelist_Core( std::move(sts), MetadataKeyValueRegex, Opts );
}


// Utility function documenting the sparse table whitelist routines for operations.
OperationArgDoc STWhitelistOpArgDoc(){
    OperationArgDoc out;

    out.name = "TableSelection";
    out.desc = "Select one or more tables."_s
               + GenericSelectionInfo("table", "rows");
    out.default_val = "all";
    out.expected = true;
    out.examples = { "last", "first", "all", "none", 
                     "#0", "#-0",
                     "!last", "!#-3",
                     "key@.*value.*", "key1@.*value1.*;key2@^value2$;first",
                     "numerous", "fewest", "more-than(5)", "!fewer-than(10)" };

    return out;
}


