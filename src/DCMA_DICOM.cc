// DCMA_DICOM.cc - A part of DICOMautomaton 2019, 2026. Written by hal clark.
//
// This file contains routines for reading and writing DICOM files.
//


#include <cstdint>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <array>
#include <vector>
#include <tuple>
#include <functional>
#include <utility>
#include <algorithm>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <cctype>

#include "YgorMisc.h"
#include "YgorLog.h"
#include "YgorString.h"
#include "YgorTime.h"

#include "Metadata.h"
#include "DCMA_DICOM.h"

namespace DCMA_DICOM {

// IEEE 754:1985 compliance checks for DICOM floating-point VRs (FL, FD, OF, OD).
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
              "DICOM requires IEEE 754:1985 32-bit single-precision floats.");
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
              "DICOM requires IEEE 754:1985 64-bit double-precision floats.");

// All non-retired transfer syntaxs require the use of little-endian byte ordering.
// The readers and writers in this file assume the host is little-endian, so verify.
static_assert(YgorEndianness::Host == YgorEndianness::Little, 
              "All non-retired DICOM transfer syntaxes require little-endian encoding.");


// Format a tag (group, element) as a hex string like "(0028,1052)" for diagnostic messages.
static std::string tag_diag(uint16_t group, uint16_t tag){
    std::ostringstream ss;
    ss << "(" << std::hex << std::setfill('0')
       << std::setw(4) << group << ","
       << std::setw(4) << tag << ")";
    return ss.str();
}

// Describe a character for diagnostic purposes. Printable chars are quoted; others shown as hex.
static std::string char_diag(char c){
    if(std::isprint(static_cast<unsigned char>(c))){
        return std::string("'") + c + "'";
    }
    std::ostringstream ss;
    ss << "0x" << std::hex << std::setfill('0')
       << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(c));
    return ss.str();
}

// Find the first character in 'val' not in 'allowed' and describe it for diagnostics.
static std::string first_invalid_char_diag(const std::string &val, const std::string &allowed){
    auto pos = val.find_first_not_of(allowed);
    if(pos != std::string::npos){
        return char_diag(val[pos]);
    }
    return "(unknown)";
}

// DICOM PS3.5 Section 6.2: Check if string contains any control characters (0x00-0x1F, 0x7F).
// Used for text VR validation (e.g., AE) where all control characters are forbidden.
// Note: NULL (0x00) is included -- while it is used as a padding character for UI/OB,
// it is not valid within text VR values themselves.
static bool has_control_char(const std::string &val){
    for(const auto c : val){
        const auto uc = static_cast<unsigned char>(c);
        if(uc <= 0x1F || uc == 0x7F) return true;
    }
    return false;
}

// DICOM PS3.5 Section 6.2: Check for control characters excluding ESC (0x1B).
// Used for VRs SH, LO, PN, UC which allow ESC for ISO/IEC 2022 escape sequences.
static bool has_control_char_except_esc(const std::string &val){
    for(const auto c : val){
        const auto uc = static_cast<unsigned char>(c);
        if((uc <= 0x1F || uc == 0x7F) && uc != 0x1B) return true;
    }
    return false;
}

// DICOM PS3.5 Section 6.2: Check for control characters excluding TAB, LF, FF, CR, ESC.
// Used for VRs ST, LT, UT which allow text formatting control characters.
static bool has_control_char_except_text(const std::string &val){
    for(const auto c : val){
        const auto uc = static_cast<unsigned char>(c);
        if((uc <= 0x1F || uc == 0x7F)
           && uc != 0x09   // TAB
           && uc != 0x0A   // LF
           && uc != 0x0C   // FF
           && uc != 0x0D   // CR
           && uc != 0x1B)  // ESC
             return true;
    }
    return false;
}

struct Node;

Node::Node() = default;

Node::Node(NodeKey key,
           std::string VR,
           std::string val) 
         : key(key),
           VR(std::move(VR)),
           val(std::move(val)) {}


bool Node::operator==(const Node &rhs) const {
    if(this == &rhs) return true;

    auto l = std::make_tuple(this->key.group, this->key.order, 
                             this->key.tag,   this->key.element);
    auto r = std::make_tuple(rhs.key.group,   rhs.key.order, 
                             rhs.key.tag,     rhs.key.element);
    return (l == r);
}

bool Node::operator!=(const Node &rhs) const {
    return !(*this == rhs);
}

bool Node::operator<(const Node &rhs) const {
    if(this == &rhs) return false;

    auto l = std::make_tuple(this->key.group, this->key.order, 
                             this->key.tag,   this->key.element);
    auto r = std::make_tuple(rhs.key.group,   rhs.key.order, 
                             rhs.key.tag,     rhs.key.element);
    return (l < r);
}

Node *
Node::emplace_child_node(Node &&n){
    Node *child_node = &( this->children.emplace_back(std::forward<Node>(n)) ); // Requires C++17.

    if( (child_node->VR == "MULTI")
    &&  (this->VR != "SQ")
    &&  (1 < this->children.size()) ){
        throw std::invalid_argument("'MULTI' nodes should only have siblings when the parent is a 'SQ' node. Refusing to continue.");
        // Note: Improper use of this tag can result in invalid DICOM files (i.e., DICOM tags can be disordered).
        // This issue can be mitigated by ensuring MULTI nodes do not have any sibling nodes (except when the parent is a sequence node).
        //
        // If this functionality is truly needed (e.g., to support DICOM modularization) then siblings not under a 'SQ'
        // node will have to be emitted in the correct monotonically-increasing DICOM group ordering. This will require
        // 'peeking' into sibling nodes when nodes are being emitted to a stream.
    }

    this->children.sort(); // Ensure the nodes are sorted as per DICOM standard.
                           // Note: This enables the DICOM write function to be const.

    return child_node;
}

// This routine writes the provided type to the provided stream. It verifies encoding, the expected length (in bytes)
// are available to be written (and there are no extras). The number of bytes written is returned.
template<class T>
uint64_t
write_to_stream( std::ostream &os,
                 const T &x,
                 uint64_t expected_length,
                 Encoding enc ){

    // Verify encoding can be handled.
    if( (enc != Encoding::ILE) 
    &&  (enc != Encoding::ELE) ){
        throw std::runtime_error("Encoding is not little-endian. This is not currently supported.");
    }

    // Verify this is a little-endian machine.
    //
    // Note: When C++20 is available we can use std::endian at compile time here instead.
    //
    // Note: We could emit DICOM files in little- or big-endian using a technique independent of the computer, but for
    //       simplicity this is not currently done.
    {
        uint16_t test { 0x01 };
        auto * first_byte = reinterpret_cast<unsigned char *>(&test);
        //static_assert((static_cast<uint16_t>(*first_byte) == test), "This computer is not little-endian. This is not supported.");
        if(static_cast<uint16_t>(*first_byte) != test){
            throw std::runtime_error("This computer is not little-endian. This is not supported.");
        }
    }

    // Ensure the correct number of bytes can be written.
    if(sizeof(T) != expected_length){
        throw std::runtime_error("Expected number of bytes does not match type size. (Is this intentional?)");
    }

    // Write the bytes.
    uint64_t written_length = 0;

    os.write(reinterpret_cast<const char *>(&x), sizeof(x));
    written_length = expected_length;

    return written_length;
}

// Explicit instantiation for writing raw bytes via std::string.
//
// Note: The contents of the string will be written in byte order, so endian conversion is not relevant for this
// routine.
template<>
uint64_t
write_to_stream( std::ostream &os,
                 const std::string &x,
                 uint64_t expected_length,
                 Encoding enc ){

    // Verify encoding can be handled.
    if( (enc != Encoding::ILE) 
    &&  (enc != Encoding::ELE) ){
        throw std::runtime_error("Encoding is not little-endian. This is not currently supported.");
    }

    // Ensure the correct number of bytes can be written.
    //
    // Note: An exact length match is required here to try catch errors between expected lengths (e.g., 2-byte VR's) and
    //       actual string contents. If the entire string should be written, supply the length of the string.
    const auto available_length = static_cast<uint64_t>(x.length());
    if(available_length != expected_length){
        throw std::runtime_error("Expected number of bytes in string does not match type size. (Is this intentional?)");
    }

    // Write the bytes.
    uint64_t written_length = 0;

    os.write(reinterpret_cast<const char *>(x.data()), available_length);
    written_length = available_length;

    return written_length;
}



// This routine emits a DICOM tag using the provided string of bytes payload. This routine handles writing the DICOM
// structure.
//
// NOTE: The payload is treated as a string of bytes and is not interpretted or adjusted for endianness.
//       All pre-processing should be taken care of before this routine is invoked.
uint64_t
emit_DICOM_tag(std::ostream &os,
               Encoding enc,
               const Node &node,   // Does not use the node's val member, to allow pre-processing.
               const std::string& val,
               bool lenient = false){

    uint64_t written_length = 0;

    written_length += write_to_stream(os, node.key.group, 2, enc);
    written_length += write_to_stream(os, node.key.tag, 2, enc);

    // With implicit encoding all tags are written in the same way.
    if(enc == Encoding::ILE){
        // Deal with sequences separately.
        if( node.VR == "SQ" ){
            if(!val.empty()){
                throw std::logic_error("'SQ' VR node passed data, but they can not have any data associated with them. (Is it intentional?)");
            }

            // Recursively emit the children to determine their lengths.
            uint64_t seq_length = 0;
            std::ostringstream seq_ss(std::ios_base::ate | std::ios_base::binary);

            for(const auto &n : node.children){
                std::ostringstream child_ss(std::ios_base::ate | std::ios_base::binary);
                const uint64_t child_length = n.emit_DICOM(child_ss, enc, false, lenient);
                const auto child_length_32 = static_cast<uint32_t>(child_length);

                seq_length += write_to_stream(seq_ss, static_cast<uint16_t>(0xFFFE), 2, enc); // group.
                seq_length += write_to_stream(seq_ss, static_cast<uint16_t>(0xE000), 2, enc); // tag.
                seq_length += write_to_stream(seq_ss, child_length_32, 4, enc);
                seq_length += write_to_stream(seq_ss, child_ss.str(), child_length, enc);
            }

            // Emit the full child lengths and serialized children.
            const auto seq_length_32 = static_cast<uint32_t>(seq_length);
            written_length += write_to_stream(os, seq_length_32, 4, enc);
            written_length += write_to_stream(os, seq_ss.str(), seq_length, enc);

        // All others.
        }else{
            const auto length    = static_cast<uint32_t>(val.length());
            const auto add_space = static_cast<uint32_t>(length % 2);
            const uint32_t full_length = (length + add_space);
            // DICOM PS3.5 Section 6.2: UI and binary/byte-stream VRs are padded with
            // NULL (0x00); all other (text) VRs are padded with SPACE (0x20).
            const bool is_null_padded = (node.VR == "UI" || node.VR == "OB" || node.VR == "OW"
                                      || node.VR == "OF" || node.VR == "OD" || node.VR == "OL"
                                      || node.VR == "OV" || node.VR == "UN" || node.VR == "SV"
                                      || node.VR == "UV");
            const auto space_char = is_null_padded ? static_cast<unsigned char>('\0')
                                                   : static_cast<unsigned char>(' ');

            written_length += write_to_stream(os, full_length, 4, enc);
            written_length += write_to_stream(os, val, length, enc);
            if(0 < add_space) written_length += write_to_stream(os, space_char, 1, enc); // Ensure length is divisible by 2.
        }

    // With explicit encoding the VR is explicitly mentioned.
    }else if(enc == Encoding::ELE){

        // Deal with sequences separately.
        if( node.VR == "SQ" ){
            if(!val.empty()){
                throw std::logic_error("'SQ' VR node passed data, but they can not have any data associated with them. (Is it intentional?)");
            }

            const uint16_t zero_16 = 0;
            written_length += write_to_stream(os, node.VR, 2, enc);
            written_length += write_to_stream(os, zero_16, 2, enc); // "Reserved" space.

            // Recursively emit the children to determine their lengths.
            uint64_t seq_length = 0;
            std::ostringstream seq_ss(std::ios_base::ate | std::ios_base::binary);

            for(const auto &n : node.children){
                std::ostringstream child_ss(std::ios_base::ate | std::ios_base::binary);
                const uint64_t child_length = n.emit_DICOM(child_ss, enc, false, lenient);
                const auto child_length_32 = static_cast<uint32_t>(child_length);

                // Emit a tag containing the length of the child.
                seq_length += write_to_stream(seq_ss, static_cast<uint16_t>(0xFFFE), 2, enc); // group.
                seq_length += write_to_stream(seq_ss, static_cast<uint16_t>(0xE000), 2, enc); // tag.
                seq_length += write_to_stream(seq_ss, child_length_32, 4, enc);
                seq_length += write_to_stream(seq_ss, child_ss.str(), child_length, enc);
            }

            // Emit the full child lengths and serialized children.
            const auto seq_length_32 = static_cast<uint32_t>(seq_length);
            written_length += write_to_stream(os, seq_length_32, 4, enc);
            written_length += write_to_stream(os, seq_ss.str(), seq_length, enc);

        // DICOM PS3.5 Section 7.1.2, Table 7.1-1: VRs not listed in Table 7.1-2 use
        // 2 reserved bytes (set to 0000H) followed by a 32-bit unsigned length field.
        // This includes: OB, OD, OF, OL, OV, OW, SV, UC, UN, UR, UT, UV.
        }else if( (node.VR == "OB")
              ||  (node.VR == "OD")
              ||  (node.VR == "OF")
              ||  (node.VR == "OL")
              ||  (node.VR == "OV")
              ||  (node.VR == "OW")
              ||  (node.VR == "SV")
              ||  (node.VR == "UC")
              ||  (node.VR == "UN")
              ||  (node.VR == "UR")
              ||  (node.VR == "UT")
              ||  (node.VR == "UV") ){
            const auto length    = static_cast<uint32_t>(val.length());
            const auto add_space = static_cast<uint32_t>(length % 2);
            const uint32_t full_length = (length + add_space);
            const uint16_t zero_16 = 0;
            // DICOM PS3.5 Section 6.2: Text VRs (UC, UR, UT) are padded with SPACE (0x20);
            // binary/byte-stream VRs (OB, OD, OF, OL, OV, OW, SV, UN, UV) with NULL (0x00).
            const auto space_char = static_cast<uint8_t>(
                (node.VR == "UC" || node.VR == "UR" || node.VR == "UT") ? 0x20 : 0x00);

            written_length += write_to_stream(os, node.VR, 2, enc);
            written_length += write_to_stream(os, zero_16, 2, enc); // "Reserved" space.
            written_length += write_to_stream(os, full_length, 4, enc);

            written_length += write_to_stream(os, val, length, enc);
            if(0 < add_space) written_length += write_to_stream(os, space_char, 1, enc); // Ensure length is divisible by 2.

        // All others do not.
        }else{
            const auto length    = static_cast<uint16_t>(val.length());
            const auto add_space = static_cast<uint16_t>(length % 2);
            const uint16_t full_length = (length + add_space);
            const auto space_char = (node.VR == "UI") ? static_cast<unsigned char>('\0')
                                                      : static_cast<unsigned char>(' ');

            written_length += write_to_stream(os, node.VR, 2, enc);
            written_length += write_to_stream(os, full_length, 2, enc);

            written_length += write_to_stream(os, val, length, enc);
            if(0 < add_space) written_length += write_to_stream(os, space_char, 1, enc); // Ensure length is divisible by 2.
        }
         
    }else{
        throw std::runtime_error("Unsupported encoding specified. Refusing to continue.");
    }

    return written_length;
}

// This routine will recursively write a DICOM file from this node and all of its children.
uint64_t Node::emit_DICOM(std::ostream &os,
                          Encoding enc,
                          bool is_root_node,
                          bool lenient) const {

    YLOGDEBUG("emit_DICOM: tag " << tag_diag(this->key.group, this->key.tag)
              << " VR='" << this->VR << "'"
              << " is_root=" << is_root_node
              << " lenient=" << lenient
              << " enc=" << (enc == Encoding::ELE ? "ELE" : "ILE"));

    // Used to search for forbidden characters.
    const std::string upper_case("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    const std::string lower_case("abcdefghijklmnopqrstuvwxyz");
    const std::string number_digits("0123456789");
    const std::string multiplicity(R"***(\)***"); // Value multiplicity separator.

    // Convenience string identifying this tag for diagnostic messages.
    const auto get_tag_str = [&](){
        return tag_diag(this->key.group, this->key.tag);
    };
    const auto get_tag_and_val_str = [&](){
        const bool is_printable = std::all_of(std::begin(this->val), std::end(this->val),
                                              [](unsigned char c) {
                                                  return std::isprint(c);
                                              });

        const auto display_val = is_printable ? this->val : "(not printable)";
        return tag_diag(this->key.group, this->key.tag) + "='" + display_val + "'";
    };


    uint64_t cumulative_length = 0;

    // If this is the root node, ignore the VR and treat it as a simple container of children.
    if(is_root_node){
        // Verify the node does not have any data associated with it. If it does, it probably indicates a logic
        // error since only children nodes should contain data.
        if(!this->val.empty()){
            throw std::logic_error("Nodes with 'SQ' VR can not have any data associated with them. (Is it intentional?)");
        }

        // Emit the DICM header before processing any nodes.
        const std::string header = std::string(128, '\0') + std::string("DICM");
        cumulative_length += write_to_stream(os, header, 132, enc);

        // Process children nodes. To generate group lengths we need to emit them in bunches.
        std::ostringstream child_ss(std::ios_base::ate | std::ios_base::binary);
        uint64_t group_length = 0;
        const auto end_child = std::end(this->children);
        for(auto child_it = std::begin(this->children); child_it != end_child; ++child_it){
            
            // Always emit the meta information header tags (group = 0x0002) with little endian explicit encoding.
            Encoding child_enc = (child_it->key.group <= 0x0002) ? Encoding::ELE : enc;
                
            // Emit this node into the temp buffer.
            group_length += child_it->emit_DICOM(child_ss, child_enc, false, lenient);

            // Evaluate whether the following node will be from a different group.
            // If so, emit the group length tag and all children in the buffer.
            const auto next_child_it = std::next(child_it);
            if( (next_child_it == end_child) 
            ||  (child_it->key.group != next_child_it->key.group) ){

                // Emit the group length tag.
                if( (child_it->key.group <= 0x0002)
                &&  (child_enc == Encoding::ELE) ){  // TODO: Should I bother with this after group 0x0002? It is deprecated...
                    Node group_length_node({child_it->key.group, 0x0000}, "UL", std::to_string(group_length));
                    cumulative_length += group_length_node.emit_DICOM(os, child_enc, false, lenient);
                }

                // Emit all the children from the buffer.
                os.write(reinterpret_cast<const char*>(child_ss.str().data()), group_length);
                cumulative_length += group_length;

                // Reset the children buffer.
                group_length = 0;
                std::ostringstream new_ss(std::ios_base::ate | std::ios_base::binary);
                child_ss.swap(new_ss);
            }
        }

    }else if( this->VR == "MULTI" ){ 
        // Not a true DICOM VR. Used to emit children without any boilerplate (cf. the 'SQ' VR).
        
        // Verify the node does not have any data associated with it. If it does, it probably indicates a logic
        // error since only children nodes should contain data.
        if(!this->val.empty()){
            throw std::logic_error("'MULTI' nodes can not have any data associated with them. (Is it intentional?) " + get_tag_and_val_str());
        }

        // Process children nodes serially, without any boilerplate or markers between children.
        for(const auto & c : this->children){
            cumulative_length += c.emit_DICOM(os, enc, false, lenient);
        }

    // Text types.
    }else if( this->VR == "CS" ){ //Code strings.
        // DICOM PS3.5 Section 6.2: CS - Code String.
        // Character repertoire: Uppercase letters, "0"-"9", SPACE, underscore "_".
        // Maximum length: 16 bytes per value.
        if(!lenient){
            // Value multiplicity embiggens the maximum permissable length, but each individual element should be <= 16 chars.
            auto tokens = SplitStringToVector(this->val,'\\','d');
            for(const auto &token : tokens){
                if(16ULL < token.length()){
                    throw std::invalid_argument("Code string is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }
            }

            const auto allowed_cs = upper_case + number_digits + multiplicity + "_ ";
            if(this->val.find_first_not_of(allowed_cs) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(this->val, allowed_cs)
                                            + " found in code string at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "SH" ){ //Short string.
        // DICOM PS3.5 Section 6.2: SH - Short String.
        // Character repertoire: Default, excluding backslash (5CH) within each value and all control characters except ESC.
        // Maximum length: 16 characters per value.
        // NOTE: Length restrictions only apply to explicit transfer syntax. Implicit encoding uses the DICOM dictionary.
        if(!lenient && enc == Encoding::ELE){
            // Value multiplicity embiggens the maximum permissible length, but each individual element should be <= 16 chars
            // and must not contain forbidden control characters (except ESC).
            auto tokens = SplitStringToVector(this->val,'\\','d');
            for(const auto &token : tokens){
                if(16ULL < token.length()){
                    throw std::runtime_error("Short string is too long at tag " + get_tag_and_val_str() + ". Consider using a longer VR. Cannot continue.");
                }
                if(has_control_char_except_esc(token)){
                    throw std::invalid_argument("Forbidden control character found in SH at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }
            }
        }else if(!lenient && enc == Encoding::ILE){
            // For implicit encoding, only check for forbidden control characters, not length.
            if(has_control_char_except_esc(this->val)){
                throw std::invalid_argument("Forbidden control character found in SH at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "LO" ){ //Long strings.
        // DICOM PS3.5 Section 6.2: LO - Long String.
        // Character repertoire: Default, excluding backslash (5CH) and all control characters except ESC, per value.
        // Maximum length: 64 characters per value. LO is multi-valued, using backslash as the VM delimiter.
        // NOTE: Length restrictions only apply to explicit transfer syntax. Implicit encoding uses the DICOM dictionary.
        if(!lenient && enc == Encoding::ELE){
            // Validate each value component separately, treating backslash as the VM delimiter.
            std::size_t start = 0;
            while(true){
                const std::size_t pos = this->val.find('\\', start);
                const std::size_t end = (pos == std::string::npos) ? this->val.size() : pos;
                const std::string value_component = this->val.substr(start, end - start);

                if(64ULL < value_component.length()){
                    throw std::runtime_error("Long string value is too long at tag " + get_tag_and_val_str() + ". Each LO value must be <= 64 characters. Cannot continue.");
                }
                if(has_control_char_except_esc(value_component)){
                    throw std::invalid_argument("Forbidden control character found in LO at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }

                if(pos == std::string::npos){
                    break;
                }
                start = pos + 1;
            }
        }else if(!lenient && enc == Encoding::ILE){
            // For implicit encoding, only check for forbidden control characters, not length.
            if(has_control_char_except_esc(this->val)){
                throw std::invalid_argument("Forbidden control character found in LO at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "ST" ){ //Short text.
        // DICOM PS3.5 Section 6.2: ST - Short Text.
        // Character repertoire: Default, excluding control characters except TAB, LF, FF, CR, ESC.
        // Maximum length: 1024 characters. Not multi-valued (backslash allowed).
        if(!lenient){
            if(1024ULL < this->val.length()){
                throw std::runtime_error("Short text is too long at tag " + get_tag_str() + ". Consider using a longer VR. Cannot continue.");
            }
            if(has_control_char_except_text(this->val)){
                throw std::invalid_argument("Forbidden control character found in ST at tag " + get_tag_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "LT" ){ //Long text.
        // DICOM PS3.5 Section 6.2: LT - Long Text.
        // Character repertoire: Default, excluding control characters except TAB, LF, FF, CR, ESC.
        // Maximum length: 10240 characters. Not multi-valued (backslash allowed).
        if(!lenient){
            if(10240ULL < this->val.length()){
                throw std::runtime_error("Long text is too long at tag " + get_tag_str() + ". Consider using a longer VR. Cannot continue.");
            }
            if(has_control_char_except_text(this->val)){
                throw std::invalid_argument("Forbidden control character found in LT at tag " + get_tag_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "UT" ){ //Unlimited text.
        // DICOM PS3.5 Section 6.2: UT - Unlimited Text.
        // Character repertoire: Default, excluding control characters except TAB, LF, FF, CR, ESC.
        // Maximum length: 2^32-2 bytes. Not multi-valued (backslash allowed).
        if(!lenient){
            if(4'294'967'294ULL < this->val.length()){
                throw std::runtime_error("Unlimited text is too long at tag " + get_tag_str() + ". Cannot continue.");
            }
            if(has_control_char_except_text(this->val)){
                throw std::invalid_argument("Forbidden control character found in UT at tag " + get_tag_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "UC" ){ //Unlimited characters.
        // DICOM PS3.5 Section 6.2: UC - Unlimited Characters.
        // Character repertoire: Default, excluding backslash (5CH) and all control characters except ESC.
        // Maximum length: 2^32-2 bytes.
        if(!lenient){
            if(4'294'967'294ULL < this->val.length()){
                throw std::runtime_error("Unlimited characters is too long at tag " + get_tag_str() + ". Cannot continue.");
            }
            if(has_control_char_except_esc(this->val)){
                throw std::invalid_argument("Forbidden control character found in UC at tag " + get_tag_str() + ". Cannot continue.");
            }
            // Note: backslash is the value multiplicity delimiter -- individual values are
            // separated by the caller, so the presence of a backslash is not itself forbidden
            // here (it is part of the encoding for multi-valued UC).
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "UR" ){ //Universal Resource Identifier or Locator (URI/URL).
        // DICOM PS3.5 Section 6.2: UR - URI/URL.
        // Character repertoire: Subset of Default required for URIs per RFC 3986 Section 2,
        //   plus trailing SPACE for padding. Leading spaces are not allowed.
        //   Not multi-valued (backslash is disallowed per RFC 3986).
        // Maximum length: 2^32-2 bytes.
        if(!lenient){
            if(4'294'967'294ULL < this->val.length()){
                throw std::runtime_error("URI is too long at tag " + get_tag_str() + ". Cannot continue.");
            }
            if(!this->val.empty() && this->val.front() == ' '){
                throw std::invalid_argument("Leading space found in UR at tag " + get_tag_str() + ". Cannot continue.");
            }
            if(this->val.find('\\') != std::string::npos){
                throw std::invalid_argument("Backslash found in UR at tag " + get_tag_str() + ". Not permitted per RFC 3986. Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);


    // Name types.
    }else if( this->VR == "AE" ){ //Application entity.
        // DICOM PS3.5 Section 6.2: AE - Application Entity.
        // Character repertoire: Default, excluding all control characters.
        // Maximum length: 16 bytes per value. Leading/trailing spaces are non-significant.
        // A value consisting solely of spaces shall not be used.
        if(!lenient){
            // AE can be multi-valued; values are separated by backslash (VM delimiter).
            std::size_t start = 0;
            while(start <= this->val.size()){
                const std::size_t end = this->val.find('\\', start);
                const std::string token = (end == std::string::npos)
                                           ? this->val.substr(start)
                                           : this->val.substr(start, end - start);

                // Empty tokens or tokens consisting solely of spaces are not permitted.
                bool has_non_space = false;
                for(char c : token){
                    if(c != ' '){
                        has_non_space = true;
                        break;
                    }
                }
                if(!token.empty() && !has_non_space){
                    throw std::invalid_argument("Application entity value consists solely of spaces at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }
                if(token.empty()){
                    throw std::invalid_argument("Empty application entity value at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }

                // Enforce 16-byte maximum per AE value.
                if(16ULL < token.size()){
                    throw std::runtime_error("Application entity value is too long (>" + std::to_string(16ULL) + " bytes) at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }

                // Control characters are forbidden.
                if(has_control_char(token)){
                    throw std::invalid_argument("Control character found in AE value at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }

                if(end == std::string::npos){
                    break;
                }
                start = end + 1;
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "PN" ){ //Person name.
        // DICOM PS3.5 Section 6.2: PN - Person Name.
        // Character repertoire: Default, excluding backslash (5CH) and all control characters except ESC.
        // Maximum length: 64 characters per component group (up to 3 groups delimited by '=').
        // Components within a group are delimited by '^' (up to 5 components).
        if(!lenient){
            // Check per-value and per-component-group limits.
            auto values = SplitStringToVector(this->val, '\\', 'd');
            for(const auto &pn_val : values){
                auto groups = SplitStringToVector(pn_val, '=', 'd');
                if(3ULL < groups.size()){
                    throw std::invalid_argument("Person name has more than 3 component groups at tag " + get_tag_str() + ". Cannot continue.");
                }
                for(const auto &group : groups){
                    if(64ULL < group.length()){
                        throw std::runtime_error("Person name component group exceeds 64 characters at tag " + get_tag_str() + ". Cannot continue.");
                    }
                }
            }
            if(has_control_char_except_esc(this->val)){
                throw std::invalid_argument("Forbidden control character found in PN at tag " + get_tag_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "UI" ){ //Unique Identifier (UID).
        // DICOM PS3.5 Section 6.2: UI - Unique Identifier.
        // Character repertoire: "0"-"9" and "." of Default Character Repertoire.
        // Maximum length: 64 bytes per UID value. Padded with trailing NULL (0x00).
        if(!lenient){
            // Value multiplicity: each UID component separated by backslash is limited to 64 bytes.
            auto uid_values = SplitStringToVector(this->val, '\\', 'd');
            for(const auto &uid_val : uid_values){
                if(64ULL < uid_val.length()){
                    throw std::runtime_error("UID is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }
            }
            const auto allowed_ui = number_digits + multiplicity + ".";
            if(this->val.find_first_not_of(allowed_ui) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(this->val, allowed_ui)
                                            + " found in UID at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }

            // Ensure there are no leading insignificant zeros.
            auto tokens = SplitStringToVector(this->val,'.','d');
            for(const auto &token : tokens){
                if( (1 < token.size()) && (token.at(0) == '0') ){
                    throw std::invalid_argument("UID contains an insignificant leading zero at tag " + get_tag_and_val_str() + ". Refusing to continue.");
                }
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);


    //Date and Time.
    }else if( this->VR == "DA" ){  //Date.
        // DICOM PS3.5 Section 6.2: DA - Date.
        // Character repertoire: "0"-"9". Format: YYYYMMDD.
        // Maximum length: 8 bytes fixed.
        //
        // Note: legacy ACR-NEMA format (YYYY.MM.DD) is stripped during pre-processing below.
        std::string digits_only(val);
        digits_only = PurgeCharsFromString(digits_only,":-");
        auto avec = SplitStringToVector(digits_only,'.','d');
        avec.resize(1);
        digits_only = Lineate_Vector(avec, "");

        if(!lenient){
            if(8ULL < digits_only.length()){
                throw std::runtime_error("Date is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
            if(digits_only.find_first_not_of(number_digits) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(digits_only, number_digits)
                                            + " found in date at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, digits_only, lenient);

    }else if( this->VR == "TM" ){  //Time.
        // DICOM PS3.5 Section 6.2: TM - Time.
        // Character repertoire: "0"-"9", ".", and SPACE (trailing padding only).
        // Format: HHMMSS.FFFFFF (2+2+2+1+6 = 13 chars, plus optional trailing space = 14).
        // Maximum length: 14 bytes.
        //
        // Note: legacy ACR-NEMA format (HH:MM:SS.frac) is stripped during pre-processing below.
        std::string digits_only(val);
        digits_only = PurgeCharsFromString(digits_only,":-");
        auto avec = SplitStringToVector(digits_only,'.','d');
        avec.resize(1);
        digits_only = Lineate_Vector(avec, "");

        if(!lenient){
            if(14ULL < digits_only.length()){
                throw std::runtime_error("Time is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
            const auto allowed_tm = number_digits + ". ";
            if(digits_only.find_first_not_of(allowed_tm) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(digits_only, allowed_tm)
                                            + " found in time at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, digits_only, lenient);

    }else if( this->VR == "DT" ){  //Date Time.
        // DICOM PS3.5 Section 6.2: DT - Date Time.
        // Character repertoire: "0"-"9", "+", "-", ".", and SPACE (trailing padding only).
        // Format: YYYYMMDDHHMMSS.FFFFFF&ZZXX. Maximum length: 26 bytes.
        //
        // Note: legacy format normalization is performed during pre-processing below.
        std::string digits_only(val);
        digits_only = PurgeCharsFromString(digits_only,":-");
        auto avec = SplitStringToVector(digits_only,'.','d');
        avec.resize(1);
        digits_only = Lineate_Vector(avec, "");

        if(!lenient){
            if(26ULL < digits_only.length()){
                throw std::runtime_error("Date-time is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
            const auto allowed_dt = number_digits + "+-. ";
            if(digits_only.find_first_not_of(allowed_dt) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(digits_only, allowed_dt)
                                            + " found in date-time at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, digits_only, lenient);

    }else if( this->VR == "AS" ){ //Age string.
        // DICOM PS3.5 Section 6.2: AS - Age String.
        // Character repertoire: "0"-"9", "D", "W", "M", "Y".
        // Length: 4 bytes fixed. Format: nnnD, nnnW, nnnM, or nnnY.
        if(!lenient){
            if(4ULL < this->val.length()){
                throw std::runtime_error("Age string is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
            const auto allowed_as = number_digits + "DWMY";
            if(this->val.find_first_not_of(allowed_as) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(this->val, allowed_as)
                                            + " found in age string at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
            if(!this->val.empty() && this->val.find_first_of("DWMY") == std::string::npos){
                throw std::invalid_argument("Age string is missing one of 'DWMY' characters at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);


    //Binary types.
    }else if( this->VR == "OB" ){ //'Other' binary string: a string of bytes that doesn't fit any other VR.
        // DICOM PS3.5 Section 6.2: OB - Other Byte.
        // An octet-stream. Insensitive to byte ordering.
        // Maximum length: 2^32-2 bytes. Padded with trailing NULL (0x00) to even length.
        if(!lenient){
            if( 4'294'967'294ULL < this->val.length() ){ // 2^32 - 2
                throw std::invalid_argument("Other byte string is too long at tag " + get_tag_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "OW" ){ //'Other word string': a string of 16bit values.
        // DICOM PS3.5 Section 6.2: OW - Other Word.
        // A stream of 16-bit words. Requires byte swapping per byte ordering.
        // Maximum length: 2^32-2 bytes. Must be a multiple of 2 bytes.
        if(!lenient){
            if( 4'294'967'294ULL < this->val.length() ){ // 2^32 - 2
                throw std::invalid_argument("Other word string is too long at tag " + get_tag_str() + ". Cannot continue.");
            }
            if( (this->val.length() % 2) != 0 ){
                throw std::invalid_argument("Other word string does not seem to contain 16-bit words at tag " + get_tag_str() + ". Cannot continue.");
            }
        }

        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "OL" ){ //'Other long': a stream of 32-bit words.
        // DICOM PS3.5 Section 6.2: OL - Other Long.
        // A stream of 32-bit words. Requires byte swapping per byte ordering.
        // Maximum length: 2^32-4 bytes. Must be a multiple of 4 bytes.
        if(!lenient){
            if( 4'294'967'292ULL < this->val.length() ){ // 2^32 - 4
                throw std::invalid_argument("Other long is too long at tag " + get_tag_str() + ". Cannot continue.");
            }
            if( (this->val.length() % 4) != 0 ){
                throw std::invalid_argument("Other long does not contain an integral number of 32-bit words at tag " + get_tag_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "OV" ){ //'Other 64-bit very long': a stream of 64-bit words.
        // DICOM PS3.5 Section 6.2: OV - Other 64-bit Very Long.
        // A stream of 64-bit words. Requires byte swapping per byte ordering.
        // Maximum length: 2^32-8 bytes. Must be a multiple of 8 bytes.
        if(!lenient){
            if( 4'294'967'288ULL < this->val.length() ){ // 2^32 - 8
                throw std::invalid_argument("Other 64-bit very long is too long at tag " + get_tag_str() + ". Cannot continue.");
            }
            if( (this->val.length() % 8) != 0 ){
                throw std::invalid_argument("Other 64-bit very long does not contain an integral number of 64-bit words at tag " + get_tag_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);


    //Numeric types that are written as a string of characters.
    }else if( this->VR == "IS" ){ //Integer string.
        // DICOM PS3.5 Section 6.2: IS - Integer String.
        // Character repertoire: "0"-"9", "+", "-", and SPACE (leading/trailing padding only).
        // Maximum length: 12 bytes per value. Range: -2^31 to 2^31-1.
        if(!lenient){
            // Overall string length limit based on encoding (IS uses short VR in ELE: 16-bit length field).
            if( ( (enc == Encoding::ELE) && (65'534ULL < this->val.length()) )
            ||  ( (enc == Encoding::ILE) && (4'294'967'295ULL < this->val.length()) ) ){
                throw std::invalid_argument("Integer string is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }

            auto tokens = SplitStringToVector(this->val,'\\','d');
            for(const auto &token : tokens){
                // Maximum length per value: 12 bytes.
                if(12ULL < token.length()){
                    throw std::invalid_argument("Integer string element is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }

                // Ensure that, if an element is present it parses as a number.
                try{
                    if(!token.empty()) [[maybe_unused]] auto r = std::stoll(token);
                }catch(const std::exception &){
                    throw std::runtime_error("Unable to convert '"_s + token + "' to IS at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }
            }

            const auto allowed_is = number_digits + multiplicity + "+-" + " ";
            if(this->val.find_first_not_of(allowed_is) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(this->val, allowed_is)
                                            + " found in integer string at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "DS" ){ //Decimal string.
        // DICOM PS3.5 Section 6.2: DS - Decimal String.
        // Character repertoire: "0"-"9", "+", "-", "E", "e", ".", and SPACE.
        // Maximum length: 16 bytes per value.
        if(!lenient){
            // Overall string length limit based on encoding (DS uses short VR in ELE: 16-bit length field).
            if( ( (enc == Encoding::ELE) && (65'534ULL < this->val.length()) )
            ||  ( (enc == Encoding::ILE) && (4'294'967'295ULL < this->val.length()) ) ){
                throw std::invalid_argument("Decimal string is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }

            auto tokens = SplitStringToVector(this->val,'\\','d');
            for(const auto &token : tokens){
                // Maximum length per decimal number: 16 bytes.
                if(16ULL < token.length()){
                    throw std::invalid_argument("Decimal string element is too long at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }

                // Ensure that if an element is present it parses as a number.
                try{
                    if(!token.empty()) [[maybe_unused]] auto r = std::stod(token);
                }catch(const std::exception &){
                    throw std::runtime_error("Unable to convert '"_s + token + "' to DS at tag " + get_tag_and_val_str() + ". Cannot continue.");
                }
            }

            const auto allowed_ds = number_digits + multiplicity + "+-eE." + " ";
            if(this->val.find_first_not_of(allowed_ds) != std::string::npos){
                throw std::invalid_argument("Invalid character " + first_invalid_char_diag(this->val, allowed_ds)
                                            + " found in decimal string at tag " + get_tag_and_val_str() + ". Cannot continue.");
            }
        }
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);


    //Numeric types that must be binary encoded.
    // Note: IEEE 754:1985 compliance for float and double is verified via static_assert at the top of this file.
    }else if( this->VR == "FL" ){ //Floating-point (IEEE 754:1985 32-bit).
        // DICOM PS3.5 Section 6.2: FL - Floating Point Single.
        // IEEE 754 binary32. Length: 4 bytes fixed.
        YLOGDEBUG("emit_DICOM: encoding FL at tag " << get_tag_and_val_str() << " val='" << this->val << "'");
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const float val_f = std::stof(this->val);
                write_to_stream(ss, val_f, 4, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for FL at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const float val_f = std::stof(this->val);
            write_to_stream(ss, val_f, 4, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "FD" ){ //Floating-point double (IEEE 754:1985 64-bit).
        // DICOM PS3.5 Section 6.2: FD - Floating Point Double.
        // IEEE 754 binary64. Length: 8 bytes fixed.
        YLOGDEBUG("emit_DICOM: encoding FD at tag " << get_tag_and_val_str() << " val='" << this->val << "'");
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const double val_d = std::stod(this->val);
                write_to_stream(ss, val_d, 8, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for FD at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const double val_d = std::stod(this->val);
            write_to_stream(ss, val_d, 8, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "OF" ){ //"Other" floating-point (IEEE 754:1985 32-bit).
        // DICOM PS3.5 Section 6.2: OF - Other Float.
        // A stream of IEEE 754 binary32 values. Maximum length: 2^32-4 bytes.
        //The value payload may contain multiple floats separated by some partitioning character.
        // For example, '1.23\2.34\0.00\25E25\-1.23'.
        YLOGDEBUG("emit_DICOM: encoding OF at tag " << get_tag_and_val_str());
        if(lenient){
            // Lenient: emit raw bytes directly, skipping token parsing.
            cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            auto tokens = SplitStringToVector(this->val, '\\', 'd');
            for(auto &token_val : tokens){
                const float val_f = std::stof(token_val);
                write_to_stream(ss, val_f, 4, enc);
            }
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "OD" ){ //"Other" floating-point double (IEEE 754:1985 64-bit).
        // DICOM PS3.5 Section 6.2: OD - Other Double.
        // A stream of IEEE 754 binary64 values. Maximum length: 2^32-8 bytes.
        //The value payload may contain multiple doubles separated by some partitioning character.
        // For example, '1.23\2.34\0.00\25E25\-1.23'.
        YLOGDEBUG("emit_DICOM: encoding OD at tag " << get_tag_and_val_str());
        if(lenient){
            // Lenient: emit raw bytes directly, skipping token parsing.
            cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            auto tokens = SplitStringToVector(this->val, '\\', 'd');
            for(auto &token_val : tokens){
                const double val_d = std::stod(token_val);
                write_to_stream(ss, val_d, 8, enc);
            }
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "SS" ){ //Signed short (16bit).
        // DICOM PS3.5 Section 6.2: SS - Signed Short.
        // Signed binary integer 16 bits. Length: 2 bytes fixed. Range: -2^15 to 2^15-1.
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const int16_t val_i = std::stoi(this->val);
                write_to_stream(ss, val_i, 2, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for SS at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const int16_t val_i = std::stoi(this->val);
            write_to_stream(ss, val_i, 2, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "US" ){ //Unsigned short (16bit).
        // DICOM PS3.5 Section 6.2: US - Unsigned Short.
        // Unsigned binary integer 16 bits. Length: 2 bytes fixed. Range: 0 to 2^16-1.
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const auto val_u = static_cast<uint16_t>(std::stoul(this->val));
                write_to_stream(ss, val_u, 2, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for US at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const auto val_u = static_cast<uint16_t>(std::stoul(this->val));
            write_to_stream(ss, val_u, 2, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "SL" ){ //Signed long (32bit).
        // DICOM PS3.5 Section 6.2: SL - Signed Long.
        // Signed binary integer 32 bits. Length: 4 bytes fixed. Range: -2^31 to 2^31-1.
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const int32_t val_l = std::stol(this->val);
                write_to_stream(ss, val_l, 4, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for SL at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const int32_t val_l = std::stol(this->val);
            write_to_stream(ss, val_l, 4, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "UL" ){ //Unsigned long (32bit).
        // DICOM PS3.5 Section 6.2: UL - Unsigned Long.
        // Unsigned binary integer 32 bits. Length: 4 bytes fixed. Range: 0 to 2^32-1.
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const uint32_t val_ul = std::stoul(this->val);
                write_to_stream(ss, val_ul, 4, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for UL at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const uint32_t val_ul = std::stoul(this->val);
            write_to_stream(ss, val_ul, 4, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "AT" ){ //Attribute tag (2x unsigned shorts representing a DICOM data tag).
        // DICOM PS3.5 Section 6.2: AT - Attribute Tag.
        // Ordered pair of 16-bit unsigned integers. Length: 4 bytes fixed.
        if(lenient){
            // Lenient: attempt conversion, fall back to raw bytes.
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                auto tokens = SplitStringToVector(this->val, '\\', 'd');
                if(tokens.size() != 2ULL){
                    throw std::runtime_error("AT token count mismatch");
                }
                for(auto &token_val : tokens){
                    const auto val_u = static_cast<uint16_t>(std::stoul(token_val));
                    write_to_stream(ss, val_u, 2, enc);
                }
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for AT at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            // Assuming the value payload contains exactly two unsigned integers, e.g., '123\234'.
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            auto tokens = SplitStringToVector(this->val, '\\', 'd');
            if(tokens.size() != 2ULL){
                throw std::runtime_error("Invalid number of integers for AT type tag at " + get_tag_and_val_str() + "; exactly 2 are needed.");
            }
            for(auto &token_val : tokens){
                const auto val_u = static_cast<uint16_t>(std::stoul(token_val));
                write_to_stream(ss, val_u, 2, enc);
            }
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "SV" ){ //Signed 64-bit very long.
        // DICOM PS3.5 Section 6.2: SV - Signed 64-bit Very Long.
        // Signed binary integer 64 bits. Length: 8 bytes fixed. Range: -2^63 to 2^63-1.
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const int64_t val_sv = std::stoll(this->val);
                write_to_stream(ss, val_sv, 8, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for SV at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const int64_t val_sv = std::stoll(this->val);
            write_to_stream(ss, val_sv, 8, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }

    }else if( this->VR == "UV" ){ //Unsigned 64-bit very long.
        // DICOM PS3.5 Section 6.2: UV - Unsigned 64-bit Very Long.
        // Unsigned binary integer 64 bits. Length: 8 bytes fixed. Range: 0 to 2^64-1.
        if(lenient){
            try{
                std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
                const uint64_t val_uv = std::stoull(this->val);
                write_to_stream(ss, val_uv, 8, enc);
                cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
            }catch(const std::exception &e){
                YLOGDEBUG("emit_DICOM: lenient fallback to raw bytes for UV at " << get_tag_and_val_str() << ": " << e.what());
                cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);
            }
        }else{
            std::ostringstream ss(std::ios_base::ate | std::ios_base::binary);
            const uint64_t val_uv = std::stoull(this->val);
            write_to_stream(ss, val_uv, 8, enc);
            cumulative_length += emit_DICOM_tag(os, enc, *this, ss.str(), lenient);
        }


    //Other types.
    }else if( this->VR == "UN" ){ //Unknown. Often needed for handling private DICOM tags.
        // DICOM PS3.5 Section 6.2: UN - Unknown.
        // An octet-stream where the encoding of the contents is unknown.
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else if( this->VR == "SQ" ){ //Sequence.
        // DICOM PS3.5 Section 6.2, Section 7.5: SQ - Sequence of Items.
        // Value is a Sequence of zero or more Items. Not data-bearing itself.
        // Verify the node does not have any data associated with it. If it does, it probably indicates a logic
        // error since only children nodes should contain data.
        if(!this->val.empty()){
            throw std::logic_error("Nodes with 'SQ' VR can not have any data associated with them at tag " + get_tag_and_val_str() + ". (Is it intentional?)");
        }

        // Recursive calls happen in the following routine.
        cumulative_length += emit_DICOM_tag(os, enc, *this, this->val, lenient);

    }else{
        throw std::runtime_error("Unknown VR type '" + this->VR + "' at tag " + get_tag_and_val_str() + ". Cannot write to tag.");
    }

    return cumulative_length;
}


bool validate_VR_conformance(const std::string &VR,
                             const std::string &val,
                             DCMA_DICOM::Encoding enc ){
    // In many cases validation can only be done when actually writing the DICOM.
    // To avoid duplicating the validation checks during emission, we simulate writing a DICOM file with the given
    // content. This results in a slow runtime, but avoids tricky code duplication.
    //
    // Note: lenient mode is intentionally *not* used here -- the purpose of this function is to check strict
    // conformance.
    YLOGDEBUG("validate_VR_conformance: VR='" << VR << "' val_length=" << val.size());
    Node root_node;
    root_node.emplace_child_node({{0x9999, 0x9999}, VR, val});

    bool valid = false;
    try{
        std::stringstream ss;
        root_node.emit_DICOM(ss, enc);
        if(ss) valid = true;
    }catch(const std::exception &){};
    return valid;
}


} // namespace DCMA_DICOM
