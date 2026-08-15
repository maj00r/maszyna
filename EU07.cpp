/*
This Source Code Form is subject to the
terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not
distributed with this file, You can
obtain one at
http://mozilla.org/MPL/2.0/.
*/
/*
MaSzyna EU07 locomotive simulator
Copyright (C) 2001-2004  Marcin Wozniak, Maciej Czapkiewicz and others
*/
/*
Authors:
MarcinW, McZapkie, Shaxbee, ABu, nbmx, youBy, Ra, winger, mamut, Q424,
Stele, firleju, szociu, hunter, ZiomalCl, OLI_EU and others
*/

#include "stdafx.h"

#include "application/application.h"
#include "utilities/Logs.h"
#include <cstdlib>
#ifdef WITHDUMPGEN
#ifdef _WIN32
#include <Windows.h>
#include <DbgHelp.h>
#include <csignal>
#include <exception>
#include <atomic>
#else
#include <execinfo.h>
#include <csignal>
#include <cstring>
#include <unistd.h>
#endif
#endif

#ifdef _MSC_VER
#pragma comment(linker, "/subsystem:windows /ENTRY:mainCRTStartup")
#endif

void export_e3d_standalone(std::string in, std::string out, int flags, bool dynamic);

#include <ctime>
#include <string>
#include <sstream>
#include <iomanip>
#include <utilities/Globals.h>

#ifdef WITHDUMPGEN
#include <exception>
static std::string live_exception_reason()
{
	std::string reason = "std::terminate (uncaught C++ exception)";
	if (std::exception_ptr ex = std::current_exception())
	{
		try
		{
			std::rethrow_exception(ex);
		}
		catch (const std::exception &e)
		{
			reason += ": ";
			reason += e.what();
		}
		catch (...)
		{
			reason += ": non-standard exception type";
		}
	}
	return reason;
}
#endif

#ifdef _WIN32
#pragma comment(lib, "Dbghelp.lib")

// set on first entry so a fault inside the handler can't recurse
static std::atomic_flag g_dump_in_progress = ATOMIC_FLAG_INIT;

static std::string module_at(const void *address)
{
	HMODULE module = nullptr;
	if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) || module == nullptr)
		return "(unknown module)";

	char path[MAX_PATH] = {0};
	GetModuleFileNameA(module, path, MAX_PATH);

	std::ostringstream oss;
	oss << path << "+0x" << std::hex << (reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(module));
	return oss.str();
}

// symbolic call stack straight into the log; needs the .pdb next to the .exe
// (present in RelWithDebInfo builds) to resolve function names and line numbers
static void write_stack_trace(std::ostream &out, const CONTEXT *context)
{
	HANDLE proc = GetCurrentProcess();
	HANDLE thread = GetCurrentThread();
	SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
	SymInitialize(proc, nullptr, TRUE);

	CONTEXT walk = *context;
	STACKFRAME64 frame = {};
	DWORD machine;
#if defined(_M_X64)
	machine = IMAGE_FILE_MACHINE_AMD64;
	frame.AddrPC.Offset = walk.Rip;
	frame.AddrFrame.Offset = walk.Rbp;
	frame.AddrStack.Offset = walk.Rsp;
#elif defined(_M_IX86)
	machine = IMAGE_FILE_MACHINE_I386;
	frame.AddrPC.Offset = walk.Eip;
	frame.AddrFrame.Offset = walk.Ebp;
	frame.AddrStack.Offset = walk.Esp;
#endif
	frame.AddrPC.Mode = AddrModeFlat;
	frame.AddrFrame.Mode = AddrModeFlat;
	frame.AddrStack.Mode = AddrModeFlat;

	char symbuf[sizeof(SYMBOL_INFO) + 256];
	auto *sym = reinterpret_cast<SYMBOL_INFO *>(symbuf);

	out << "stack:\n";
	for (int depth = 0; depth < 64; ++depth)
	{
		if (!StackWalk64(machine, proc, thread, &frame, &walk, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr) || frame.AddrPC.Offset == 0)
			break;

		sym->SizeOfStruct = sizeof(SYMBOL_INFO);
		sym->MaxNameLen = 255;
		DWORD64 symoff = 0;
		const char *name = SymFromAddr(proc, frame.AddrPC.Offset, &symoff, sym) ? sym->Name : "??";

		IMAGEHLP_LINE64 line;
		line.SizeOfStruct = sizeof(line);
		DWORD lineoff = 0;
		if (SymGetLineFromAddr64(proc, frame.AddrPC.Offset, &lineoff, &line))
			out << "  " << name << "  (" << line.FileName << ":" << line.LineNumber << ")\n";
		else
			out << "  " << name << "\n";
	}
	SymCleanup(proc);
}

static void write_crash_dump(EXCEPTION_POINTERS *info, const char *reason)
{
	if (g_dump_in_progress.test_and_set())
		return;

	SYSTEMTIME st;
	GetLocalTime(&st);

	std::ostringstream stamp;
	stamp << std::setw(4) << std::setfill('0') << st.wYear << "-" << std::setw(2) << std::setfill('0') << st.wMonth << "-" << std::setw(2) << std::setfill('0') << st.wDay << "_" << std::setw(2)
	      << std::setfill('0') << st.wHour << "-" << std::setw(2) << std::setfill('0') << st.wMinute << "-" << std::setw(2) << std::setfill('0') << st.wSecond;

	std::string dmp = "crash_" + stamp.str() + ".dmp";

	HANDLE hFile = CreateFileA(dmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (hFile != INVALID_HANDLE_VALUE)
	{
		MINIDUMP_EXCEPTION_INFORMATION dumpInfo;
		dumpInfo.ThreadId = GetCurrentThreadId();
		dumpInfo.ExceptionPointers = info;
		dumpInfo.ClientPointers = FALSE;

		MINIDUMP_TYPE dumpType = MINIDUMP_TYPE(MiniDumpWithFullMemory | MiniDumpWithHandleData | MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules | MiniDumpWithIndirectlyReferencedMemory |
		                                       MiniDumpWithFullMemoryInfo | MiniDumpWithTokenInformation);

		MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, dumpType, info ? &dumpInfo : nullptr, nullptr, nullptr);
		CloseHandle(hFile);
	}

	std::ostringstream diag;
	diag << "\n===== CRASH =====\n";
	diag << "time   : " << stamp.str() << "\n";
	diag << "reason : " << reason << "\n";
	if (info && info->ExceptionRecord)
	{
		diag << "code   : 0x" << std::hex << std::uppercase << info->ExceptionRecord->ExceptionCode << std::dec << "\n";
		diag << "address: " << info->ExceptionRecord->ExceptionAddress << "\n";
		diag << "module : " << module_at(info->ExceptionRecord->ExceptionAddress) << "\n";
		if (info->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && info->ExceptionRecord->NumberParameters >= 2)
		{
			diag << "access : " << (info->ExceptionRecord->ExceptionInformation[0] == 0 ? "read" : info->ExceptionRecord->ExceptionInformation[0] == 1 ? "write" : "execute") << " at 0x" << std::hex
			     << info->ExceptionRecord->ExceptionInformation[1] << std::dec << "\n";
		}
	}
	diag << "thread : " << GetCurrentThreadId() << "\n";
	diag << "dump   : " << dmp << "\n";
	if (info && info->ContextRecord)
		write_stack_trace(diag, info->ContextRecord);
	diag << "=================\n";

	CrashLog(diag.str());

	std::string msg = "Simulator crash occured :(\n\nReason: " + std::string(reason) + "\nDump saved: " + dmp;
	MessageBoxA(nullptr, msg.c_str(), "Simulator crashed :(", MB_ICONERROR);
}

// terminate/abort/purecall/invalid-parameter never reach the SEH filter, so
// synthesize the exception info from the current register context
static void write_crash_dump_here(const char *reason)
{
	CONTEXT ctx;
	RtlCaptureContext(&ctx);

	EXCEPTION_RECORD rec = {};
	rec.ExceptionCode = 0xE0000001; // application-defined "software raised" code
#if defined(_M_X64)
	rec.ExceptionAddress = reinterpret_cast<PVOID>(ctx.Rip);
#elif defined(_M_IX86)
	rec.ExceptionAddress = reinterpret_cast<PVOID>(ctx.Eip);
#else
	rec.ExceptionAddress = _ReturnAddress();
#endif

	EXCEPTION_POINTERS ep;
	ep.ExceptionRecord = &rec;
	ep.ContextRecord = &ctx;

	write_crash_dump(&ep, reason);
}

LONG WINAPI CrashHandler(EXCEPTION_POINTERS *ExceptionInfo)
{
	write_crash_dump(ExceptionInfo, "unhandled exception (SEH)");
	return EXCEPTION_EXECUTE_HANDLER;
}

static void crash_terminate()
{
	write_crash_dump_here(live_exception_reason().c_str());
	std::_Exit(3);
}

static void crash_purecall()
{
	write_crash_dump_here("pure virtual function call");
	std::_Exit(3);
}

static void crash_invalid_parameter(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int, uintptr_t)
{
	write_crash_dump_here("CRT invalid parameter");
	std::_Exit(3);
}

static void crash_signal(int sig)
{
	write_crash_dump_here(sig == SIGABRT ? "abort() / SIGABRT" : "fatal signal");
	std::_Exit(3);
}

static void install_crash_handlers()
{
	// Keep the OS from popping its own "app stopped working" box and stealing the crash.
	SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);

	SetUnhandledExceptionFilter(CrashHandler);

	// Reserve stack for the handler so a stack-overflow crash can still dump.
	ULONG guarantee = 64 * 1024;
	SetThreadStackGuarantee(&guarantee);

	std::set_terminate(crash_terminate);
	_set_purecall_handler(crash_purecall);
	_set_invalid_parameter_handler(crash_invalid_parameter);

	// Don't let abort() throw up the CRT dialog before our handler runs.
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
	signal(SIGABRT, crash_signal);
}

#elif defined(WITHDUMPGEN) // ---------------------------------------- POSIX

// There is no minidump on Linux; instead we log a symbolic backtrace to the
// game log and then re-raise the signal so the OS can still drop a core file
// (if ulimit -c allows it) for post-mortem debugging.
static void crash_backtrace(const char *reason)
{
	void *frames[64];
	int count = backtrace(frames, 64);
	char **symbols = backtrace_symbols(frames, count);

	std::ostringstream diag;
	diag << "\n===== CRASH =====\n";
	diag << "reason : " << reason << "\n";
	diag << "backtrace:\n";
	for (int i = 0; i < count; ++i)
		diag << "  " << (symbols ? symbols[i] : "??") << "\n";
	diag << "=================\n";

	CrashLog(diag.str());

	// also to stderr, using the async-signal-safe variant
	backtrace_symbols_fd(frames, count, STDERR_FILENO);

	free(symbols);
}

static void crash_signal(int sig)
{
	crash_backtrace(strsignal(sig));

	// restore default disposition and re-raise to generate a core dump
	signal(sig, SIG_DFL);
	raise(sig);
}

static void crash_terminate()
{
	crash_backtrace(live_exception_reason().c_str());
	std::abort();
}

static void install_crash_handlers()
{
	signal(SIGSEGV, crash_signal);
	signal(SIGABRT, crash_signal);
	signal(SIGFPE, crash_signal);
	signal(SIGILL, crash_signal);
	signal(SIGBUS, crash_signal);
	std::set_terminate(crash_terminate);
}

#endif

int main(int argc, char *argv[])
{
#ifdef WITHDUMPGEN
	install_crash_handlers();
#endif
	// init start timestamp
	Global.startTimestamp = std::chrono::steady_clock::now();

	// quick short-circuit for standalone e3d export
	if (argc == 6 && std::string(argv[1]) == "-e3d")
	{
		std::string in(argv[2]);
		std::string out(argv[3]);
		int flags = std::stoi(std::string(argv[4]));
		int dynamic = std::stoi(std::string(argv[5]));
		export_e3d_standalone(in, out, flags, dynamic);
	}
	else
	{
		try
		{
			auto result{Application.init(argc, argv)};
			if (result == 0)
			{
				result = Application.run();
				Application.exit();
			}
		}
		catch (std::bad_alloc const &Error)
		{
			CrashLog("\n===== FATAL =====\nreason : memory allocation failure: " + std::string(Error.what()) + "\n=================");
		}
		catch (std::exception const &Error)
		{
			CrashLog("\n===== FATAL =====\nreason : " + std::string(Error.what()) + "\n=================");
#ifdef _WIN32
			MessageBoxA(nullptr, ("Simulator crash occured :(\n" + std::string(Error.what())).c_str(), "Simulator crashed :(", MB_ICONERROR);
#endif
		}
	}
#ifndef _WIN32
	fflush(stdout);
	fflush(stderr);
#endif
	std::_Exit(0); // skip destructors, there are ordering errors which causes segfaults
}
