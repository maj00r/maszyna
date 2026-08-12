// Exercises the real cParser, built against the stubs in tests/stubs.
//
// The cases that matter here are the ones no amount of reading catches: loops in the
// simulator that end only because the parser eventually reports eof() or stops being
// ok(). A parser that keeps returning empty tokens without ever moving the stream state
// hangs the loader instead of finishing the file.

// parser.h expects the precompiled header to have been pulled in ahead of it
#include "stdafx.h"
#include "utilities/parser.h"

#include <cstdio>
#include <filesystem>

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

std::filesystem::path scratchDir() {
    auto const directory{std::filesystem::temp_directory_path() / "eu07-parser-tests"};
    std::filesystem::create_directories(directory);
    return directory;
}

// writes a file with the exact bytes given, no newline translation
std::string writeFile(std::string const &Name, std::string const &Content) {
    auto const path{scratchDir() / Name};
    std::ofstream file(path, std::ios_base::binary | std::ios_base::trunc);
    file.write(Content.data(), static_cast<std::streamsize>(Content.size()));
    file.close();
    return path.string();
}

std::string scratchPrefix() {
    return scratchDir().string() + "/";
}

// mirrors the loop in Mover.cpp LoadFIZ: pull line sized tokens until the parser stops
// being ok. returns the tokens read, or reports a hang once the guard trips
std::vector<std::string> readLikeLoadFIZ(std::string const &Name, bool &Hung, int const Guard = 100000) {
    cParser parser(Name, cParser::buffer_FILE, scratchPrefix());
    std::vector<std::string> lines;
    Hung = false;
    while (parser.ok()) {
        lines.emplace_back(parser.getToken<std::string>(false, "\n\r"));
        if (static_cast<int>(lines.size()) > Guard) {
            Hung = true;
            break;
        }
    }
    return lines;
}

// mirrors the loop in scenery_scanner.cpp: pull tokens until the parser reports eof
int readUntilEof(std::string const &Name, bool &Hung, int const Guard = 100000) {
    cParser parser(Name, cParser::buffer_FILE, scratchPrefix());
    parser.expandIncludes = false;
    int count = 0;
    Hung = false;
    while (false == parser.eof()) {
        parser.getTokens();
        if (++count > Guard) {
            Hung = true;
            break;
        }
    }
    return count;
}

void testLoadFizStyleLoopTerminates() {
    struct testcase {
        std::string name;
        std::string content;
        std::string label;
    };
    // every shape a physics file has been seen to end with. the include split over
    // separate lines is how the shipped .fiz files reference their parameter sets
    std::vector<testcase> const cases{
        {"plain.fiz", "Param. M=1\r\nWheels: D=1\r\n", "trailing newline"},
        {"nonewline.fiz", "Param. M=1\r\nWheels: D=1", "no trailing newline"},
        {"comment.fiz", "Param. M=1\r\n// a comment\r\n", "comment then newline"},
        {"comment_eof.fiz", "Param. M=1\r\n// a comment runs to the end", "comment running into eof"},
        {"block_eof.fiz", "Param. M=1\r\n/* unterminated block", "unterminated block comment"},
        {"quote_eof.fiz", "Param. M=1\r\nname \"unterminated", "unterminated quote"},
        {"empty.fiz", "", "empty file"},
        {"blanks.fiz", "\r\n\r\n\r\n", "nothing but blank lines"},
        {"crlf_only.fiz", "\r\n", "a single blank line"},
    };

    for (auto const &testcase : cases) {
        writeFile(testcase.name, testcase.content);
        bool hung = false;
        readLikeLoadFIZ(testcase.name, hung);
        check(false == hung, "LoadFIZ style loop terminates: " + testcase.label);
    }
}

void testLoadFizStyleLoopTerminatesWithIncludes() {
    // es64f4_2.fiz in the shipped data is exactly this: an include, its file name and
    // the terminating end, each on its own line
    writeFile("inner.inc", "Param. M=1\r\nWheels: D=1\r\nend");
    writeFile("outer.fiz", "include\r\ninner.inc\r\nend\r\n");
    bool hung = false;
    auto const lines{readLikeLoadFIZ("outer.fiz", hung)};
    check(false == hung, "LoadFIZ style loop terminates over a multi line include");

    // the same shape, with the include file ending in a comment and no newline
    writeFile("inner2.inc", "Param. M=1\r\n38\t0\t//trailing comment");
    writeFile("outer2.fiz", "include\r\ninner2.inc\r\nend\r\n");
    readLikeLoadFIZ("outer2.fiz", hung);
    check(false == hung, "LoadFIZ style loop terminates when the include ends in a comment");

    // a missing include must not stall the outer file either
    writeFile("outer3.fiz", "include\r\nthere-is-no-such-file.inc\r\nend\r\n");
    readLikeLoadFIZ("outer3.fiz", hung);
    check(false == hung, "LoadFIZ style loop terminates over a missing include");
}

void testScannerStyleLoopTerminates() {
    writeFile("scan.scn", "node -1 -1 none track normal 1 0 0 0\r\nendnode\r\n");
    bool hung = false;
    readUntilEof("scan.scn", hung);
    check(false == hung, "scanner style loop terminates");

    writeFile("scan_empty.scn", "");
    readUntilEof("scan_empty.scn", hung);
    check(false == hung, "scanner style loop terminates on an empty file");
}

void testMissingFileIsNotOk() {
    cParser parser("no-such-file-8a3f.scn", cParser::buffer_FILE, scratchPrefix());
    check(false == parser.ok(), "a parser over a missing file is not ok");
    check(parser.getToken<std::string>() == "", "a parser over a missing file yields no tokens");
}

void testTokensSurviveTheEndOfFile() {
    // the last token must come out whole even without a separator behind it
    writeFile("tail.txt", "alpha beta gamma");
    cParser parser("tail.txt", cParser::buffer_FILE, scratchPrefix());
    check(parser.getToken<std::string>() == "alpha", "first token");
    check(parser.getToken<std::string>() == "beta", "middle token");
    check(parser.getToken<std::string>() == "gamma", "last token without trailing separator");
    check(parser.getToken<std::string>() == "", "nothing past the end");
}

void testCommentsAndQuotes() {
    writeFile("mix.txt", "one // skipped\r\ntwo /* skipped */ three \"quoted phrase\" four");
    cParser parser("mix.txt", cParser::buffer_FILE, scratchPrefix());
    check(parser.getToken<std::string>() == "one", "token before a line comment");
    check(parser.getToken<std::string>() == "two", "token after a line comment");
    check(parser.getToken<std::string>() == "three", "token after a block comment");
    check(parser.getToken<std::string>() == "quoted phrase", "quoted phrase arrives whole");
    check(parser.getToken<std::string>() == "four", "token after a quoted phrase");
}

void testNumericTokens() {
    writeFile("numbers.txt", "1.5 -2.25 42 none");
    cParser parser("numbers.txt", cParser::buffer_FILE, scratchPrefix());
    check(parser.getToken<float>() == 1.5f, "float token");
    check(parser.getToken<double>() == -2.25, "negative double token");
    check(parser.getToken<int>() == 42, "int token");
    check(parser.getToken<float>() == 0.f, "unconvertible token yields zero");
}

void testLineCounting() {
    writeFile("lines.txt", "a\r\nb\r\nc");
    cParser parser("lines.txt", cParser::buffer_FILE, scratchPrefix());
    parser.getToken<std::string>();
    check(parser.Line() == 1, "line counter starts at one");
    parser.getToken<std::string>();
    check(parser.Line() == 2, "line counter follows the second token");
    parser.getToken<std::string>();
    check(parser.Line() == 3, "line counter follows the third token");
}

} // namespace

int main() {
    std::printf("cParser over the real implementation\n");

    testLoadFizStyleLoopTerminates();
    testLoadFizStyleLoopTerminatesWithIncludes();
    testScannerStyleLoopTerminates();
    testMissingFileIsNotOk();
    testTokensSurviveTheEndOfFile();
    testCommentsAndQuotes();
    testNumericTokens();
    testLineCounting();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures != 0;
}
