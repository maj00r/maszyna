// char_source has to supply characters exactly as the std::istream operations it
// replaced did, including the stream state they leave behind: several parser loops in
// the simulator terminate on eof() or fail() and nothing else. every case below is
// checked against a real std::istream performing the same sequence.

#include "utilities/charsource.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool const Condition, std::string const &Label) {
    ++g_checks;
    if (false == Condition) {
        ++g_failures;
        std::printf("  FAILED: %s\n", Label.c_str());
    }
}

std::string stateOf(std::istream const &Stream) {
    return std::string(Stream.eof() ? "eof " : "") + (Stream.fail() ? "fail " : "") +
           (Stream.bad() ? "bad " : "") + (Stream.good() ? "good" : "");
}

// drives both a char_source and a plain istream through the same sequence of operations
// and requires the returned characters and the resulting stream state to agree
enum class op { peek, get };

void compareWithIstream(std::string const &Data, std::vector<op> const &Sequence,
                        std::string const &Label) {
    std::istringstream mine(Data);
    std::istringstream reference(Data);
    char_source source(mine);

    for (std::size_t i = 0; i < Sequence.size(); ++i) {
        int got = 0;
        int want = 0;
        if (Sequence[i] == op::peek) {
            got = source.peek();
            want = reference.peek();
        } else {
            got = source.get();
            want = reference.get();
        }
        check(got == want, Label + ": step " + std::to_string(i) + " returned " +
                               std::to_string(got) + ", istream returned " + std::to_string(want));
        check(stateOf(mine) == stateOf(reference),
              Label + ": step " + std::to_string(i) + " left state [" + stateOf(mine) +
                  "], istream left [" + stateOf(reference) + "]");
    }
}

std::vector<op> repeated(op const Op, int const Count) {
    return std::vector<op>(static_cast<std::size_t>(Count), Op);
}

std::vector<op> alternating(int const Count) {
    std::vector<op> sequence;
    for (int i = 0; i < Count; ++i) {
        sequence.push_back(i % 2 == 0 ? op::peek : op::get);
    }
    return sequence;
}

void testAgainstIstream() {
    // reading exactly to the end, then past it: this is where eofbit and failbit appear
    compareWithIstream("ab", repeated(op::get, 5), "get past end");
    compareWithIstream("ab", repeated(op::peek, 5), "peek past end");
    compareWithIstream("ab", alternating(10), "peek and get mixed");
    compareWithIstream("", repeated(op::get, 3), "get on empty data");
    compareWithIstream("", repeated(op::peek, 3), "peek on empty data");
    compareWithIstream("", alternating(4), "mixed on empty data");
    // peek must not consume, so an interleaved run has to stay aligned
    compareWithIstream("track 1 2 3", alternating(30), "mixed over longer data");
    // a peek reporting eof followed by a get has to add failbit, matching istream
    compareWithIstream("x", {op::get, op::peek, op::get}, "peek then get at end");
    compareWithIstream("x", {op::get, op::get, op::peek}, "get then peek at end");
    // embedded zero bytes and high bytes must survive as unsigned values
    compareWithIstream(std::string("a\0b", 3), repeated(op::get, 4), "embedded zero byte");
    compareWithIstream("\xC5\xBA\xC3\xB3", repeated(op::get, 5), "utf-8 payload");
    compareWithIstream("\xFF\xFE", repeated(op::get, 3), "high bytes are not EOF");
}

void testHighByteIsNotEof() {
    // 0xFF sign extended would compare equal to EOF and truncate every file containing it
    std::istringstream data(std::string("\xFF", 1));
    char_source source(data);
    check(source.peek() == 0xFF, "peek returns 0xFF as 255, not EOF");
    check(source.get() == 0xFF, "get returns 0xFF as 255, not EOF");
    check(source.get() == EOF, "get reports EOF after the payload");
}

void testDetachedSourceSuppliesNothing() {
    char_source source;
    check(false == source.attached(), "default constructed source is not attached");
    check(source.peek() == EOF, "detached peek reports EOF");
    check(source.get() == EOF, "detached get reports EOF");
}

void testFailedStreamSuppliesNothing() {
    // a missing include file produces exactly this: the parser keeps reading from it
    std::ifstream missing("this-file-does-not-exist-4e6f70ff.txt", std::ios_base::binary);
    check(true == missing.fail(), "the probe stream really did fail to open");
    char_source source(missing);
    check(false == source.attached(), "source over a failed stream is not attached");
    check(source.peek() == EOF, "failed stream peek reports EOF");
    check(source.get() == EOF, "failed stream get reports EOF");
}

void testBumpAfterPeek() {
    // bump is only ever called once peek has reported a character
    std::istringstream data("ab");
    char_source source(data);
    check(source.peek() == 'a', "peek sees the first character");
    check(source.bump() == 'a', "bump consumes the peeked character");
    check(source.peek() == 'b', "peek sees the second character");
    check(source.bump() == 'b', "bump consumes the second character");
    check(source.peek() == EOF, "peek reports EOF once data runs out");
}

void testAttachResets() {
    std::istringstream first("a");
    std::istringstream second("bc");
    char_source source(first);
    check(source.get() == 'a', "reads from the first stream");
    check(source.get() == EOF, "first stream runs out");
    source.attach(second);
    check(source.attached(), "re-attached to a usable stream");
    check(source.get() == 'b', "reads from the second stream after re-attaching");
}

void testStateDrivenLoopTerminates() {
    // deserialize_map and the scenery scanner spin on these predicates; a source that
    // never moves the stream state would hang the loader instead of ending the file
    std::istringstream data("a b c");
    char_source source(data);
    int guard = 0;
    while (false == data.eof() && guard < 1000) {
        source.peek();
        source.get();
        ++guard;
    }
    check(guard < 1000, "a loop ending on eof() terminates");

    std::istringstream other("a b c");
    char_source othersource(other);
    guard = 0;
    while (false == other.fail() && guard < 1000) {
        othersource.get();
        ++guard;
    }
    check(guard < 1000, "a loop ending on fail() terminates");
}

} // namespace

int main() {
    std::printf("char_source against std::istream\n");

    testAgainstIstream();
    testHighByteIsNotEof();
    testDetachedSourceSuppliesNothing();
    testFailedStreamSuppliesNothing();
    testBumpAfterPeek();
    testAttachResets();
    testStateDrivenLoopTerminates();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures != 0;
}
