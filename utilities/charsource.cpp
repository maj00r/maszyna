/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/

#include "utilities/charsource.h"

char_source::char_source( std::istream &Stream ) {

    attach( Stream );
}

void char_source::attach( std::istream &Stream ) {

    m_stream = &Stream;
    // a stream which failed to open has a buffer, but reading it is pointless and the
    // failed state has to stay visible to whoever asks the parser whether it is usable
    m_buffer = ( true == Stream.fail() ? nullptr : Stream.rdbuf() );
}

// std::istream operations open with a sentry which fails on a stream that is no longer
// good, and raises failbit when it does. reads past the end therefore report eof first
// and only pick up failbit on the attempt after that, which is what ends the parser
// loops written against ok()
bool char_source::readable() {

    if( true == m_stream->good() ) { return true; }

    m_stream->setstate( std::ios_base::failbit );
    return false;
}

int char_source::peek() {

    if( m_buffer == nullptr ) { return EOF; }
    if( false == readable() ) { return EOF; }

    auto const character { m_buffer->sgetc() };
    if( character == std::char_traits<char>::eof() ) {
        // std::istream::peek raises eofbit alone
        m_stream->setstate( std::ios_base::eofbit );
        return EOF;
    }
    return character;
}

int char_source::get() {

    if( m_buffer == nullptr ) { return EOF; }
    if( false == readable() ) { return EOF; }

    auto const character { m_buffer->sbumpc() };
    if( character == std::char_traits<char>::eof() ) {
        // std::istream::get raises failbit as well when it extracts nothing
        m_stream->setstate( std::ios_base::eofbit | std::ios_base::failbit );
        return EOF;
    }
    return character;
}

int char_source::bump() {

    if( m_buffer == nullptr ) { return EOF; }
    return m_buffer->sbumpc();
}

bool char_source::attached() const {

    return m_buffer != nullptr;
}
