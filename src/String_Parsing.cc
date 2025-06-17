//String_Parsing.cc - A part of DICOMautomaton 2021. Written by hal clark.

#include <algorithm>
#include <vector>
#include <string>
#include <optional>
#include <cstdint>
#include <sstream>
#include <limits>
#include <optional>
#include <codecvt>

#include "YgorString.h"
#include "YgorMath.h"
#include "YgorMisc.h"
#include "YgorLog.h"

#include "String_Parsing.h"

template <class T>
std::optional<T>
get_as(const std::string &in){

    if( !Is_String_An_X<T>(in) ){
        return std::optional<T>();
    }else{
        return std::make_optional(stringtoX<T>(in));
    }
}
template std::optional<uint8_t    > get_as(const std::string &);
template std::optional<uint16_t   > get_as(const std::string &);
template std::optional<uint32_t   > get_as(const std::string &);
template std::optional<uint64_t   > get_as(const std::string &);
template std::optional<int8_t     > get_as(const std::string &);
template std::optional<int16_t    > get_as(const std::string &);
template std::optional<int32_t    > get_as(const std::string &);
template std::optional<int64_t    > get_as(const std::string &);
template std::optional<float      > get_as(const std::string &);
template std::optional<double     > get_as(const std::string &);
template std::optional<std::string> get_as(const std::string &);


void array_to_string(std::string &s, const std::array<char, 2048> &a){
    s.clear();
    for(const auto &c : a){
        if(c == '\0') break;
        s.push_back(c);
    }
    return;
}

std::string array_to_string(const std::array<char, 2048> &a){
    std::string s;
    array_to_string(s,a);
    return s;
}

void string_to_array(std::array<char, 2048> &a, const std::string &s){
    a.fill('\0');
    for(size_t i = 0; (i < s.size()) && ((i+1) < a.size()); ++i){
        a[i] = s[i];
        //a[i+1] = '\0';
    }
    return;
}

std::array<char, 2048> string_to_array(const std::string &s){
    std::array<char, 2048> a;
    string_to_array(a,s);
    return a;
}


// Remove characters so that the argument can be inserted like '...' or "..." on command line without interfering with
// the quotes.
//
// Note that this does *NOT* protect against shell expansion within "..." quotes.
std::string escape_for_quotes(std::string s){
    // Remove unprintable characters and newlines.
    // We also remove quotes of either kind to ensure the result can be embedded in either kind of quotes.
    const auto rem = [](unsigned char c){
        return (   !std::isprint(c)
                || (c == '\'')
                || (c == '"')
                || (c == '\n')
                || (c == '\r') );
    };
    s.erase( std::remove_if(std::begin(s), std::end(s), rem),
             std::end(s) );
    return s;
}


template <class T>
std::string to_string_max_precision(T x){
    std::stringstream ss;
    const auto defaultprecision = ss.precision();
    ss.precision(std::numeric_limits<T>::max_digits10);
    ss << x;
    ss.precision(defaultprecision);
    return ss.str();
}
template std::string to_string_max_precision(float x);
template std::string to_string_max_precision(double x);


std::vector<parsed_function>
parse_functions(const std::string &in, 
                char escape_char,
                char func_sep_char,
                int64_t parse_depth){

    std::vector<parsed_function> out;
    const bool debug = false;

    // Parse function statements respecting quotation and escapes, converting
    //
    //   `f1(x, "arg, text\, or \"escaped\" sequence", 1.23); f2("tex\t", 2.\34)`
    //
    // into the following parsed function name and parameter tokens:
    //
    //  Function:
    //    Function name: 'f1'
    //    Parameter 1:   'x'
    //    Parameter 2:   'arg, text, or "escaped" sequence'
    //    Parameter 3:   '1.23'
    //  Function:
    //    Function name: 'f2'
    //    Parameter 1:   'text'  <--- n.b. the 't' is normal.
    //    Parameter 2:   '2.\34' <--- n.b. escaping only works inside a quotation
    //
    // Note that nested functions *are* supported. They use syntax like:
    //
    //    `parent(x, y, z){ child1(a, b, c); child2(d, e, f) }`
    //
    // Also note that quotations can be used to avoid nested function parse issues,
    // which will convert
    //
    //    `f1(x, "f2(a,b,c)")`
    //
    // into:
    //
    //  Function:
    //    Function name: 'f1'
    //    Parameter 1:   'x'
    //    Parameter 2:   'f2(a,b,c)'
    //

    {
        const auto clean_string = [](const std::string &in){
            return Canonicalize_String2(in, CANONICALIZE::TRIM_ENDS);
        };
        const auto clean_function_name = [](const std::string &in){
            std::string out = in;
            out.erase( std::remove_if( std::begin(out), std::end(out),
                                       [](unsigned char c){
                                           return !(std::isalnum(c) || (c == '_')); 
                                       }),
                       std::end(out));
            return out;
        };
        
        parsed_function pshtl;
        std::string shtl;
        std::vector<char> close_quote; // e.g., '"' or '\''.
        std::vector<char> close_paren; // e.g., ')' or ']'.
        
        const auto end = std::end(in);
        for(auto c_it = std::begin(in); c_it != end; ++c_it){

            // Behaviour inside a curly parenthesis (i.e., nested children functions).
            //
            // Note that nested functions are parsed recursively.
            if( !close_paren.empty()
            &&  (close_paren.front() == '}') ){

                // Behaviour inside a quote.
                if( !close_quote.empty() ){

                    // Handle escapes by immediately advancing to the next char.
                    if( *c_it == escape_char ){
                        shtl.push_back(*c_it);
                        ++c_it;
                        if(c_it == end){
                            throw std::invalid_argument("Escape character present, but nothing to escape");
                        }
                        shtl.push_back(*c_it);
                        continue;

                    // Close the quote, discarding the quote symbol.
                    }else if( *c_it == close_quote.back() ){
                        shtl.push_back(*c_it);
                        close_quote.pop_back();
                        continue;

                    }else{
                        shtl.push_back(*c_it);
                        continue;
                    }

                }else{

                    // Open a quote.
                    if( (*c_it == '\'') || (*c_it == '"') ){
                        close_quote.push_back(*c_it);
                        shtl.push_back(*c_it);
                        continue;

                    // Increase the nesting depth.
                    }else if( *c_it == '{' ){
                        close_paren.push_back('}');
                        shtl.push_back(*c_it);
                        continue;

                    // Close a parenthesis.
                    }else if( *c_it == close_paren.back() ){
                        close_paren.pop_back();

                        // Parse the children.
                        if(close_paren.empty()){
                            if(out.empty()){
                                throw std::invalid_argument("No parent function available to append child to");
                            }
                            if(!out.back().children.empty()){
                                throw std::invalid_argument("Function already contains one or more nested functions");
                            }
                            shtl = clean_string(shtl);
                            if(!shtl.empty()){
                                out.back().children = parse_functions(shtl, escape_char, func_sep_char, parse_depth + 1);
                            }
                            shtl.clear();

                        // Note: we only drop the top-level parenthesis. Pass all others through for recursive parsing.
                        }else{
                            shtl.push_back(*c_it);
                        }
                        continue;

                    }else{
                        shtl.push_back(*c_it);
                        continue;
                    }
                }

            // Behaviour inside a quote.
            }else if( !close_quote.empty() ){

                // Handle escapes by immediately advancing to the next char.
                if( *c_it == escape_char ){
                    ++c_it;
                    if(c_it == end){
                        throw std::invalid_argument("Escape character present, but nothing to escape");
                    }
                    shtl.push_back(*c_it);
                    continue;

                // Close the quote, discarding the quote symbol.
                }else if( *c_it == close_quote.back() ){
                    close_quote.pop_back();
                    continue;

                }else{
                    shtl.push_back(*c_it);
                    continue;
                }

            // Behaviour inside a parenthesis (i.e., the 'parameters' part of a function).
            }else if( !close_paren.empty()
                  &&  ( (close_paren.back() == ')') || (close_paren.back() == ']') ) ){

                // Open a quote and discard the quoting character.
                if( (*c_it == '\'') || (*c_it == '"') ){
                    close_quote.push_back(*c_it);
                    continue;

                // Close a parenthesis, which should also complete the function.
                }else if( *c_it == close_paren.back() ){
                    // Complete the final argument.
                    //
                    // Note that to support disregarding a trailing comma, we disallow empty arguments.
                    shtl = clean_string(shtl);
                    if(!shtl.empty()){
                        pshtl.parameters.emplace_back();
                        pshtl.parameters.back().raw = shtl;
                    }
                    shtl.clear();

                    // Reset the shtl function.
                    out.emplace_back(pshtl);
                    pshtl = parsed_function();

                    close_paren.pop_back();
                    continue;

                // Handle arg -> arg transition, discarding the comma character.
                }else if(*c_it == ','){
                    // Note that to support disregarding a trailing comma, we disallow empty arguments.
                    shtl = clean_string(shtl);
                    if(!shtl.empty()){
                        pshtl.parameters.emplace_back();
                        pshtl.parameters.back().raw = shtl;
                    }
                    shtl.clear();
                    continue;

                }else{
                    shtl.push_back(*c_it);
                    continue;
                }

            // Behaviour outside of a quote or parenthesis.
            }else{

                // Handle function name -> args transition.
                // Open a parentheses and discard the character.
                if( (*c_it == '(') || (*c_it == '[') ){
                    if(*c_it == '('){
                        close_paren.push_back(')');
                    }else if(*c_it == '['){
                        close_paren.push_back(']');
                    }

                    // Assign function name.
                    shtl = clean_string(shtl);
                    shtl = clean_function_name(shtl);
                    if(shtl.empty()){
                        throw std::invalid_argument("Function names cannot be empty");
                    }
                    if(!pshtl.name.empty()){
                        throw std::invalid_argument("Refusing to overwrite existing function name");
                    }
                    pshtl.name = shtl;
                    shtl.clear();
                    continue;

                // Parse nested children.
                //
                // Note: we drop the top-level parenthesis here.
                }else if( *c_it == '{' ){
                    close_paren.push_back('}');
                    continue;

                // Handle function -> function transitions.
                }else if(*c_it == func_sep_char){
                    shtl = clean_string(shtl);
                    if(!shtl.empty()){
                        throw std::invalid_argument("Disregarding characters between functions");
                    }
                    shtl.clear();

                }else{
                    shtl.push_back(*c_it);
                    continue;
                }
            }

        }

        if( !pshtl.name.empty()
        ||  !pshtl.parameters.empty() ){
            throw std::invalid_argument("Incomplete function statement: terminate function by opening/closing scope");
        }

        if(!close_quote.empty()){
            throw std::invalid_argument("Imbalanced quote");
        }
        if(!close_paren.empty()){
            throw std::invalid_argument("Imbalanced parentheses");
        }

    }

    // Print the AST for debugging.
    if( debug
    &&  (parse_depth == 0) ){
        std::function<void(const std::vector<parsed_function> &, std::string)> print_ast;
        print_ast = [&print_ast](const std::vector<parsed_function> &pfv, std::string depth){
            for(auto &pf : pfv){
               YLOGINFO(depth << "name = '" << pf.name << "'");
               for(auto &p : pf.parameters){
                   YLOGINFO(depth << "  parameter: '" << p.raw << "'");
               }
               YLOGINFO(depth << "  children: " << pf.children.size());
               if(!pf.children.empty()){
                   print_ast(pf.children, depth + "    ");
               }
            }
        };
        print_ast(out, "");
    }

    if(out.empty()) throw std::invalid_argument("Unable to parse function from input");

    // Post-process parameters.
    for(auto &pf : out){
       for(auto &p : pf.parameters){
           if(p.raw.empty()) continue;

           // Try extract the number, ignoring any suffixes.
           try{
               const auto x = std::stod(p.raw);
               p.number = x;
           }catch(const std::exception &){ }

           // If numeric, assess whether the (optional) suffix indicates something important.
           if((p.raw.size()-1) == p.raw.find('x')) p.is_fractional = true;
           if((p.raw.size()-1) == p.raw.find('%')) p.is_percentage = true;

           if( p.is_fractional
           &&  p.is_percentage ){
                throw std::invalid_argument("Parameter cannot be both a fraction and a percentage");
           }
       }
    }
    
    return out;
}

std::vector<parsed_function>
retain_only_numeric_parameters(std::vector<parsed_function> pfs){
    for(auto &pf : pfs){
        pf.parameters.erase(std::remove_if(pf.parameters.begin(), pf.parameters.end(),
                            [](const function_parameter &fp){
                                return !(fp.number);
                            }),
        pf.parameters.end());

        if(!pf.children.empty()){
            pf.children = retain_only_numeric_parameters(pf.children);
        }
    }
    return pfs;
}


// Parser for number lists.
std::vector<double>
parse_numbers(const std::string &split_chars, const std::string &in){
    std::vector<std::string> split;
    split.emplace_back(in);
    for(const auto& c : split_chars){
        split = SplitVector(split, c, 'd');
    }

    std::vector<double> numbers;
    for(const auto &w : split){
       try{
           const auto x = std::stod(w);
           numbers.emplace_back(x);
       }catch(const std::exception &){ }
    }
    return numbers;
}

// Wide string narrowing.
std::string convert_wstring_to_string(const std::wstring &wstr){
    std::wstring_convert<std::codecvt_utf8<wchar_t>> l_conv;
    return l_conv.to_bytes(wstr);
}

