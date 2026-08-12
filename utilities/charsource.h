/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <istream>

/////////////////////////////////////////////////////////////////////////////////////////////////////
// char_source -- character supply for the parser

// reading a character through std::istream costs a sentry construction, which the token
// loop cannot afford at the volume of scenery data. characters are taken from the stream
// buffer directly instead, and the stream state is moved by hand to match what the
// istream operations would have set: callers end their loops on eof() and fail(), so a
// stream that never reaches those states turns them into endless loops
class char_source {

public:
// constructors
    char_source() = default;
    explicit char_source( std::istream &Stream );

// methods
    // attaches to provided stream. a stream which failed to open supplies no characters
    void attach( std::istream &Stream );
    // returns next character without consuming it, or EOF. raises eofbit at end of data
    int peek();
    // consumes and returns next character, or EOF. raises eofbit and failbit at end of data
    int get();
    // consumes and returns next character. only legal after peek() reported a character
    int bump();
    // true if a stream is attached and its buffer can be read
    bool attached() const;

private:
// methods
    // mirrors the sentry std::istream operations open with, failbit included
    bool readable();

// members
    std::istream *m_stream { nullptr };
    std::streambuf *m_buffer { nullptr };
};
