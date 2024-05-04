#!/usr/bin/env python
# coding:UTF-8
__all__ = ['header', 'Fore', 'RESET']

import ctypes
import os
import re


def enable_vt_processing():
    _ansi_256_enabled = {
        'ANSICON',
        'COLORTERM',
        'ConEmuANSI',
        'PYCHARM_HOSTED',
        'TERM',
        'TERMINAL_EMULATOR',
        'TERM_PROGRAM',
        'WT_SESSION'
    }
    _ansi_256_enabled.intersection_update(set(list(os.environ.keys())))
    if not any(_ansi_256_enabled):
        if os.name == 'nt':
            STD_OUTPUT_HANDLE = -11
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            if handle == -1:
                return False
            mode = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if not ctypes.windll.kernel32.SetConsoleMode(handle, mode):
                return False
    return True


ansi_256_enabled = bool(enable_vt_processing())
RESET = '\033[0m'


class Fore:
    CYAN = '\033[38;5;14m' if ansi_256_enabled else '\033[36m'
    RED = '\033[38;5;9m' if ansi_256_enabled else '\033[31m'


def adjust_ansi_codes(_text):
    if not ansi_256_enabled:
        ansi_reg_codes = {
            '\033[38;5;9m': '\033[31m',
            '\033[38;5;15m': '\033[37m',
            '\033[38;5;8m': '\033[90m'
        }
        ansi_256_codes = re.findall(r'\x1B[@-_][0-?]*[ -/]*[@-~]', _text)
        ansi_256_unique = []
        for i in ansi_256_codes:
            if not ansi_reg_codes.get(i) or i in ansi_256_unique:
                continue
            ansi_256_unique.append(i)
            _text = _text.replace(i, ansi_reg_codes[i])
    return _text


def header():
    _header = f'''
\033[38;5;9m                       .    .             .             ...               
             ,.....    ;     .           ...          ..;;;..   .....          
          .;i         .s     .s.        ..;..       ,ji&OOoq;.   ......       
        .jil          SS      s$.       .;;;.       g!'     %S.   ;;;;;;;.     
      .ll$l          .$S      .s$.     .;;$;;.     ;$        $s  ;b@$$SSijc. 
    ..7l$l           .$S       ;$S     .;$@$;.    .s$        S$. ;K       `s;  
   ..77$S           .;$*       ;$$    .;$$*$$;.   .S$        S$; :B         '   
 ..ps$$$S           .;$l       i$$    .$$* *$$.   ;S$        S$; ;;$           
 ;sXSf$$s           .s$$      .$$$   .;$*   *$;.  ;S$        S$;  ;;$          
 si`7f$$s           sS$$*...+*$$$$  .;$*******$;. .S$        S$.   ;;$.        
 F  .>$$s          jyXF====##@@$$$  .S$##333##$S.  ;$s       S$;*.  ;*$    .   
    .$$$S        ;iZ ?&ttfPPPPQ$$$ ..$$       $$.. .lSs      S$.zS. .!$$   .   
    .s$$$s     .sZZ  ???       *$s .S$         $S.   $$S,   ;S$ sHl:.;x$$  :   
    .ss$$Ss...;d$Z   ?li       *$. .S           S.    ;$@@sS$$; 'S$$$XX#$$.!   
    ..ss@@@#GS$Z/    ;ll      .S.  sS           Ss     .SSSSS.     \$$$@@$;S.  
     .ssS&&#$ff/     .;l      ;s   s.            ;      s;  s       .\X$$$Ss;  
      ssSSSXF/`       ;!      s    ;.            ;      ;   ;          xXSsZ,  
      .sSS27.          ;      *    ;             ;      ;   ;          .l5Zz. 
      .sSS4            ;      ;    ;             ;      ;   ;           .xXx;. 
     ..sSi;            .      .    ;             ;      ;   ;           .\VS;. 
     ..SSl             .      .    .             .      ;   .            .SS;. 
     .;Sl;             .      .    .             .      .   .            .IS;. 
    ..Sl..........\033[38;5;15m................................................\033[38;5;9m.........S;. 
    .;Sl;         \033[38;5;15m`##############################################`        \033[38;5;9m.lS. 
    .SSl          \033[38;5;15m`##############################################`         \033[38;5;9mll. 
    .Sl.          \033[38;5;15m`###\033[38;5;9m/ __/\033[38;5;15m###\033[38;5;9m/ /\033[38;5;15m###########\033[38;5;9m/ __// /\033[38;5;15m#############`         \033[38;5;9m.l. 
    .Si...........\033[38;5;15m`##\033[38;5;9m/ _/ / _  // _ `// -_) \ \ / __// -_)/ _ `/\033[38;5;15m#`\033[38;5;9m..........l. 
    .l1           \033[38;5;15m`#\033[38;5;9m/___/ \_,_/ \_, / \__//___/ \__/ \__/ \_, /\033[38;5;15m##`         \033[38;5;9m.l. 
    .l;           \033[38;5;15m`############\033[38;5;9m/___/\033[38;5;15m#####################\033[38;5;9m/___/\033[38;5;15m###`         \033[38;5;9m.i. 
    .l;           \033[38;5;15m`##############################################`         \033[38;5;9m.i. 
     ;.           \033[38;5;15m`##############################################`         \033[38;5;9m.i  
     ..```````````\033[38;5;15m````````````````````````````````````\033[38;5;8mcrypt0lith\033[38;5;15m``\033[38;5;9m`````````..  
     .             Chaos-Based Edge Adaptive Steganography Tool             .  
{RESET}
    '''
    return adjust_ansi_codes(_header)


if __name__ == '__main__':
    print(header())
