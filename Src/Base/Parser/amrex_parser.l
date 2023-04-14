%option noyywrap nodefault
%option never-interactive
%{
#include "AMReX_Parser_Y.H"
#include "amrex_parser.tab.h"
#ifdef _WIN32
#define YY_NO_UNISTD_H
#endif
%}

%option nounput
%option noinput
%option prefix="amrex_parser"

 /* Tokens NUMBER, SYMBOL, F1, POW, F2, etc. are defined in amrex_parser.y. */
 /* Types PARSER_F1_SQRT, PARSER_F1_EXP, etc. are defined in AMReX_Parser_y.H. */

 /* Used later to define NUMBER */
EXP	([Ee][-+]?[0-9]+)

%%

"+" |
"-" |
"*" |
"/" |
"=" |
"," |
"<" |
">" |
";" |
"(" |
")"     { return amrex_parsertext[0]; } /* simply pass through */

 /* amrex_parserlval is union type defined in amrex_parser.tab.h that is generated
  * by bison with amrex_parser.y */

"sqrt"        { amrex_parserlval.f1 = amrex::PARSER_F1_SQRT;      return F1; }
"exp"         { amrex_parserlval.f1 = amrex::PARSER_F1_EXP;       return F1; }
"log"         { amrex_parserlval.f1 = amrex::PARSER_F1_LOG;       return F1; }
"log10"       { amrex_parserlval.f1 = amrex::PARSER_F1_LOG10;     return F1; }
"sin"         { amrex_parserlval.f1 = amrex::PARSER_F1_SIN;       return F1; }
"cos"         { amrex_parserlval.f1 = amrex::PARSER_F1_COS;       return F1; }
"tan"         { amrex_parserlval.f1 = amrex::PARSER_F1_TAN;       return F1; }
"asin"        { amrex_parserlval.f1 = amrex::PARSER_F1_ASIN;      return F1; }
"acos"        { amrex_parserlval.f1 = amrex::PARSER_F1_ACOS;      return F1; }
"atan"        { amrex_parserlval.f1 = amrex::PARSER_F1_ATAN;      return F1; }
"sinh"        { amrex_parserlval.f1 = amrex::PARSER_F1_SINH;      return F1; }
"cosh"        { amrex_parserlval.f1 = amrex::PARSER_F1_COSH;      return F1; }
"tanh"        { amrex_parserlval.f1 = amrex::PARSER_F1_TANH;      return F1; }
"abs"         { amrex_parserlval.f1 = amrex::PARSER_F1_ABS;       return F1; }
"fabs"        { amrex_parserlval.f1 = amrex::PARSER_F1_ABS;       return F1; }
"floor"       { amrex_parserlval.f1 = amrex::PARSER_F1_FLOOR;     return F1; }
"ceil"        { amrex_parserlval.f1 = amrex::PARSER_F1_CEIL;      return F1; }
"comp_ellint_1" { amrex_parserlval.f1 = amrex::PARSER_F1_COMP_ELLINT_1; return F1; }
"comp_ellint_2" { amrex_parserlval.f1 = amrex::PARSER_F1_COMP_ELLINT_2; return F1; }
"**"          { amrex_parserlval.f2 = amrex::PARSER_F2_POW;       return POW;}
"^"           { amrex_parserlval.f2 = amrex::PARSER_F2_POW;       return POW;}
">="          { amrex_parserlval.f2 = amrex::PARSER_F2_GEQ;       return GEQ;}
"<="          { amrex_parserlval.f2 = amrex::PARSER_F2_LEQ;       return LEQ;}
"=="          { amrex_parserlval.f2 = amrex::PARSER_F2_EQ;        return EQ;}
"!="          { amrex_parserlval.f2 = amrex::PARSER_F2_NEQ;       return NEQ;}
"and"         { amrex_parserlval.f2 = amrex::PARSER_F2_AND;       return AND;}
"or"          { amrex_parserlval.f2 = amrex::PARSER_F2_OR;        return OR;}
"pow"         { amrex_parserlval.f2 = amrex::PARSER_F2_POW;       return F2; }
"heaviside"   { amrex_parserlval.f2 = amrex::PARSER_F2_HEAVISIDE; return F2; }
"jn"          { amrex_parserlval.f2 = amrex::PARSER_F2_JN;        return F2; }
"min"         { amrex_parserlval.f2 = amrex::PARSER_F2_MIN;       return F2; }
"max"         { amrex_parserlval.f2 = amrex::PARSER_F2_MAX;       return F2; }
"fmod"        { amrex_parserlval.f2 = amrex::PARSER_F2_FMOD;      return F2; }
"if"          { amrex_parserlval.f3 = amrex::PARSER_F3_IF;        return F3; }

 /* We use SYMBOL to hold variables and constants */
[a-zA-Z_][a-zA-Z0-9_]*  { amrex_parserlval.s = amrex::parser_makesymbol(amrex_parsertext); return SYMBOL; }

 /* Number */
[0-9]+"."[0-9]*{EXP}? |
"."?[0-9]+{EXP}?  { amrex_parserlval.d = std::atof(amrex_parsertext);   return NUMBER; }

 /* Special characters */
"//".*
[ \t]   /* ignore white space */
\\\n    /* ignore line continuation */
"\n"    { return EOL; }

 /* everything else */
.	{ amrex_parsererror("Unknown character %c", *amrex_parsertext); }

%%
