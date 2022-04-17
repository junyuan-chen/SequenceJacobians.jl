function _isleadlag(ex)
    iscall(ex, :lead) || iscall(ex, :lag) || return false
    N = length(ex.args)
    N in (2, 3) || return false
    ex.args[2] isa Symbol || return false
    N == 3 && return ex.args[3] isa Int
    return true
end

function _islessleadlag(ex1, ex2)
    n1, n2 = ex1.args[2], ex2.args[2]
    n1 == n2 || return isless(n1, n2)
    f1, f2 = ex1.args[1], ex2.args[1]
    f1 == f2 || return f1 == :lag
    s1 = length(ex1.args) == 2 ? 1 : ex1.args[3]
    s2 = length(ex2.args) == 2 ? 1 : ex2.args[3]
    return f1 == :lag ? isless(s2, s1) : isless(s1, s2)
end

function _parseins(exprs)
    ins = :(())
    for x in exprs
        if x isa Symbol
            push!(ins.args, Expr(:quote, x))
        elseif _isleadlag(x)
            f = x.args[1]
            v = x.args[2]
            if length(x.args) == 2
                push!(ins.args, Expr(:call, f, Expr(:quote, v)))
            else
                push!(ins.args, Expr(:call, f, Expr(:quote, v), x.args[3]))
            end
        else
            error("unrecognized expression in arguments")
        end
    end
    return ins
end

function _parseouts(expr)
    outs = :(())
    if isexpr(expr, :tuple)
        for x in expr.args
            if x isa Symbol
                push!(outs.args, Expr(:quote, x))
            else
                error("return statement must only contain variable names")
            end
        end
    elseif expr isa Symbol
        push!(outs.args, Expr(:quote, expr))
    else
        error("return statement must only contain variable names")
    end
    return outs
end

function _walkbodysimple(x, leadlags, outs)
    if _isleadlag(x)
        push!(leadlags, x)
        return Symbol(x.args...)
    elseif isexpr(x, :return)
        push!(outs, x.args)
    end
    return x
end

macro simple(args...)
    narg = length(args)
    narg == 0 && throw(ArgumentError("no argument is found for @simple"))
    func = args[end]
    @capture(func, function f_(ins__) body_ end) ||
        throw(ArgumentError("the last argument of @simple must be a function block"))
    kwargs = narg > 1 ? args[1:end-1] : ()
    leadlags = Set{Expr}()
    outs = []
    body = postwalk(x->_walkbodysimple(x, leadlags, outs), body)
    leadlags = sort!([leadlags...], lt=_islessleadlag)
    fargs = (ins..., ntuple(i->Symbol(leadlags[i].args...), length(leadlags))...)
    ins = _parseins((ins..., leadlags...))
    isempty(outs) && error("explicit return statement is not found")
    # If multiple return statements exist, only consider the first one found
    outs = _parseouts(outs[1][1])
    blkf = Symbol(f, :_block)
    return quote
        function $(esc(f))($(map(esc, fargs)...))
            $(esc(body))
        end
        function $(esc(blkf))()
            block($(esc(f)), $ins, $outs; ($(map(esc, kwargs)...),)...)
        end
    end
end

function _walkbodyimplicit(x, leadlags, args, parsedrets)
    if _isleadlag(x)
        push!(leadlags, x)
        return Symbol(x.args...)
    elseif isexpr(x, :return)
        parsedret = _parsereturnimplicit(x.args[1], args)
        push!(parsedrets, parsedret)
        ret = parsedret[1]
        return :(return $ret)
    end
    return x
end

function _parseargsimplicit(exprs)
    args = []
    vals = []
    for x in exprs
        if isexpr(x, :kw)
            push!(args, x.args[1])
            push!(vals, x.args[1]=>x.args[2])
        else
            error("unrecognized expression in arguments")
        end
    end
    return args, vals
end

function _parsereturnimplicit(rawret, args)
    msg = "return statement is not in the required format"
    isexpr(rawret, :tuple) && length(rawret.args) === 3 || error(msg)
    retexpr = :(())
    outs = Symbol[]
    outquote = :(())
    out = rawret.args[1]
    if out isa Symbol
        out in args || push!(retexpr.args, out)
        push!(outs, out)
        push!(outquote.args, Expr(:quote, out))
    elseif isexpr(out, :tuple)
        for x in out.args
            if x isa Symbol
                out in args || push!(retexpr.args, x)
                push!(outs, x)
                push!(outquote.args, Expr(:quote, x))
            else
                error(msg)
            end
        end
    else
        error(msg)
    end
    tars = Pair{Symbol}[]
    tarquote = :(())
    rawtar = rawret.args[2]
    if rawtar isa Symbol
        push!(retexpr.args, rawtar)
        push!(tars, rawtar=>0)
        push!(tarquote.args, Expr(:quote, rawtar))
    elseif isexpr(rawtar, :(=))
        v = rawtar.args[1]
        push!(retexpr.args, v)
        push!(tars, v=>rawtar.args[2])
        push!(tarquote.args, Expr(:quote, v))
    elseif isexpr(rawtar, :tuple)
        for x in rawtar.args
            if x isa Symbol
                push!(retexpr.args, x)
                push!(tars, x=>0)
                push!(tarquote.args, Expr(:quote, x))
            elseif isexpr(x, :(=))
                v = x.args[1]
                push!(retexpr.args, v)
                push!(tars, v=>x.args[2])
                push!(tarquote.args, Expr(:quote, v))
            else
                error(msg)
            end
        end
    else
        error(msg)
    end
    solver = rawret.args[3]
    if solver isa Symbol || isexpr(solver, :.)
        return retexpr, outs, outquote, tars, tarquote, solver
    elseif isexpr(solver, :call) && length(solver.args)==1
        return retexpr, outs, outquote, tars, tarquote, solver.args[1]
    else
        error(msg)
    end
end

function _parsevals(vals, outs)
    calis = Pair{Symbol}[]
    inits = Pair{Symbol}[]
    for val in vals
        push!(val[1] in outs ? inits : calis, val)
    end
    return calis, inits
end

macro implicit(args...)
    narg = length(args)
    narg == 0 && throw(ArgumentError("no argument is found for @implicit"))
    func = args[end]
    @capture(func, function f_(ins__) body_ end) ||
        throw(ArgumentError("the last argument of @implicit must be a function block"))
    kwargs = narg > 1 ? args[1:end-1] : ()
    args, vals = _parseargsimplicit(ins)
    leadlags = Set{Expr}()
    parsedrets = []
    body = postwalk(x->_walkbodyimplicit(x, leadlags, args, parsedrets), body)
    leadlags = sort!([leadlags...], lt=_islessleadlag)
    fargs = (args..., ntuple(i->Symbol(leadlags[i].args...), length(leadlags))...)
    argquote = _parseins((args..., leadlags...))
    isempty(parsedrets) && error("explicit return statement is not found")
    # If multiple return statements exist, only consider the first one found
    _, outs, outquote, tars, tarquote, solver = parsedrets[1]
    calis, inits = _parsevals(vals, outs)
    inquote = _parseins(ntuple(i->calis[i][1], length(calis)))
    blkf = Symbol(f, :_block)
    return quote
        function $(esc(f))($(map(esc, fargs)...))
            $(esc(body))
        end
        function $(esc(blkf))()
            b = block($(esc(f)), $argquote, $tarquote; ($(map(esc, kwargs)...),)...)
            return block(b, $inquote, $outquote,
                $(esc(calis)), $(esc(inits)), $(esc(tars));
                solver=$(esc(solver)), ($(map(esc, kwargs)...),)...)
        end
    end
end
