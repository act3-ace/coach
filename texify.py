#!/usr/bin/env python3

import os, re, argparse

latex_header = r'''
\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{textcomp}
\usepackage{listingsutf8}
\usepackage{hyperref}
\usepackage[dvipsnames]{xcolor}
\definecolor{darkgreen}{rgb}{0,0.5,0}
\definecolor{lightblue}{rgb}{0.2,0.5,1}
\hypersetup{colorlinks=true, linkcolor=blue}
\lstset{
	numbers=left,
	upquote=true,
	breaklines=true,
	tabsize=4,
	showstringspaces=false,
	showspaces=false,
	breakatwhitespace=true,
	<SYNTAX_HIGHLIGHTING>
}
\begin{document}
\tableofcontents
\newpage
'''

styles = {
'default': r'''
	basicstyle=\ttfamily\scriptsize,
	keywordstyle=\ttfamily,
	commentstyle=\ttfamily\color{darkgreen},
	stringstyle=\ttfamily\color{blue},
''',
'dark': r'''
	backgroundcolor=\ttfamily\color{black},
	basicstyle=\ttfamily\color{white}\scriptsize,
	keywordstyle=\ttfamily,
	commentstyle=\ttfamily\color{green},
	stringstyle=\ttfamily\color{lightblue},
''' # xterm-mode
}

# Governs syntax highlighting
file_types = { 
	'.py': 'Python',
	 '.c': 'C',
	 '.d': 'C',
	 '.m': 'Matlab',
	 '.r': 'R',
	 '.sh': 'bash',
	 '.bash': 'bash',
	 '.cpp': 'C++',
	 '.cc': 'C++',
	 '.pl': 'Perl',
	 '.tex': 'TeX',
	 '.f': 'Fortran',
	 '.for': 'Fortran',
	 '.ftn': 'Fortran',
	 '.f90': 'Fortran',
	 '.f95': 'Fortran',
	 '.f03': 'Fortran',
	 '.f08': 'Fortran',
	 '.csh': 'csh',
	 '.ksh': 'ksh',
	 '.lisp': 'lisp',
	 '.lsp': 'lisp',
	 '.cl': 'lisp',
	 '.l': 'lisp',
	 '.scm': 'lisp',
	 '.go': 'Go',
	 '.hs': 'Haskell',
	 '.lhs': 'Haskell',
	 '.bat': 'command.com',
	 '.awk': 'Awk',
}

def main() -> None:
	parser = argparse.ArgumentParser(usage='%s [-d DIR] [-i extension ...]\n' % __file__
		+ 'example: %s -d ./src -i foo.m -i makefile .c .d .py\n\n' % __file__,
		description='Will search under DIR for all source files with the specified file extensions, and compile them into a LaTeX file.')
	parser.add_argument('--dir', '-d', help='root directory under which to search', default='.')
	parser.add_argument('--include', '-i', action='append', help="Explicitly include a file even if it doesn't match the extension list", default=[])
	parser.add_argument('--style', default='default', choices=styles.keys(), help='Changes syntax highlighting, etc.')
	parser.add_argument('extension', nargs='+', help='Only files with these extensions will be included (leading dot optional)')
	args = parser.parse_args()

	# Permit valid extensions to be input with or without the dot
	args.extension = [ a if ('.' == a[0]) else '.%s' % a
		for a in args.extension ]

	# Make relative to base path, escape underscores
	def format_path(path: str) -> str:
		if path == args.dir: return '/'
		assert (path[0:len(args.dir)+1] == args.dir + '/') or (path[0:len(args.dir)+1] == args.dir + '\\')
		return re.sub('_', r'\_', path[len(args.dir)+1:])

	# Print single file
	def dumpsrc(dirpath: str, fname: str) -> str:
		path = '%s/%s' % (dirpath, fname)
		escaped = format_path(path)
		print(r'\subsection[%s]{%s}' % (os.path.basename(escaped), escaped))
		ext = os.path.splitext(f)[1]
		if ext in file_types:
			s = r'\lstinputlisting[language=%s]{%s}' % (file_types[ext], path)
		else:
			s = r'\lstinputlisting{%s}' % path
		return '%s\n%s\n' % (s, r'\newpage')

	def print_header() -> None:
		s = latex_header.replace(r'<SYNTAX_HIGHLIGHTING>', styles[args.style].strip(), 1)
		print(s.strip())

	print_header()

	dirs = {dirpath:fnames for dirpath, _, fnames in os.walk(args.dir)}
	includes = { os.path.realpath(f):f for f in args.include }
	for dirpath in sorted(dirs):
		fnames = dirs[dirpath]
		src = sorted([f for f in fnames 
			if (os.path.splitext(f)[1] in args.extension) or (os.path.realpath(f) in includes)])
		if 0 == len(src): continue

		print(r'\section{%s}' % format_path(dirpath))
		for f in src:
			print(dumpsrc(dirpath, f))

			# Don't include files twice just because they're explicitly included with -i
			f = os.path.realpath(f)
			if f in includes:
				del includes[f]

	# Any explicitly included files that weren't already covered (i.e. those outside args.path)
	if len(includes):
		print(r'\section{Miscellaneous}')
		for _,f in includes.items():
			f = args.dir + '/' + os.path.relpath(f, args.dir)
			print(dumpsrc(os.path.dirname(f), os.path.basename(f)))

	print(r'\end{document}')

main()
