function _isleadlag(ex)
    iscall(ex, :lead) || iscall(ex, :lag) || return false
    N = length(ex.args)
    N in (2, 3) || return false
    ex.args[2] isa Symbol || return false
    N == 3 && return ex.args[3] isa Int
    return true
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
    leadlags = (leadlags...,)
    fargs = (ins..., map(x->Symbol(x.args...), leadlags)...)
    ins = _parseins((ins..., leadlags...))
    isempty(outs) && error("explicit return statement is not found")
    # If multiple return statements exist, only consider the first one
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

function _parsetarnames(expr)
    msg = "return statement is not in the required format"
    isexpr(expr, :tuple) && length(expr.args) === 3 || error(msg)
    tars = expr.args[2]
    if tars isa Symbol
        return tars
    elseif isexpr(tars, :(=))
        return tars.args[1]
    elseif isexpr(tars, :tuple)
        outs = :(())
        for x in tars.args
            if x isa Symbol
                push!(outs.args, x)
            elseif isexpr(x, :(=))
                push!(outs.args, x.args[1])
            else
                error(msg)
            end
        end
        return outs
    else
        error(msg)
    end
end

function _walkbodyimplicit(x, leadlags, rawrets, tarnames)
    if _isleadlag(x)
        push!(leadlags, x)
        return Symbol(x.args...)
    elseif isexpr(x, :return)
        push!(rawrets, x.args[1])
        x = _parsetarnames(x.args[1])
        push!(tarnames, x)
        return :(return $x)
    end
    return x
end

function _parseargsimplicit(exprs)
    ins = []
    vals = []
    for x in exprs
        if isexpr(x, :kw)
            push!(ins, x.args[1])
            push!(vals, x.args[1]=>x.args[2])
        else
            error("unrecognized expression in arguments")
        end
    end
    return ins, vals
end

function _parsereturnimplicit(rawrets)
    msg = "return statement is not in the required format"
    outs = Symbol[]
    outexprs = :(())
    rawout = rawrets.args[1]
    if rawout isa Symbol
        push!(outs, rawout)
        push!(outexprs.args, Expr(:quote, rawout))
    elseif isexpr(rawout, :tuple)
        for x in rawout.args
            x isa Symbol ? push!(outs, x) : error(msg)
            push!(outexprs.args, Expr(:quote, x))
        end
    else
        error(msg)
    end
    tars = Pair{Symbol}[]
    rawtar = rawrets.args[2]
    if rawtar isa Symbol
        push!(tars, rawtar=>0)
    elseif isexpr(rawtar, :(=))
        push!(tars, rawtar.args[1]=>rawtar.args[2])
    elseif isexpr(rawtar, :tuple)
        for x in rawtar.args
            if x isa Symbol
                push!(tars, x=>0)
            elseif isexpr(x, :(=))
                push!(tars, x.args[1]=>x.args[2])
            else
                error(msg)
            end
        end
    else
        error(msg)
    end
    solver = rawrets.args[3]
    if solver isa Symbol || isexpr(solver, :.)
        return outs, outexprs, tars, solver
    elseif isexpr(solver, :call) && length(solver.args)==1
        return outs, outexprs, tars, solver.args[1]
    else
        error(msg)
    end
end

function _parsevals(vals, outs)
    inits = Pair{Symbol}[]
    calis = Pair{Symbol}[]
    for val in vals
        push!(val[1] in outs ? inits : calis, val)
    end
    return inits, calis
end

macro implicit(args...)
    narg = length(args)
    narg == 0 && throw(ArgumentError("no argument is found for @implicit"))
    func = args[end]
    @capture(func, function f_(ins__) body_ end) ||
        throw(ArgumentError("the last argument of @implicit must be a function block"))
    kwargs = narg > 1 ? args[1:end-1] : ()
    ins, vals = _parseargsimplicit(ins)
    leadlags = Set{Expr}()
    rawrets = []
    tarnames = []
    body = postwalk(x->_walkbodyimplicit(x, leadlags, rawrets, tarnames), body)
    leadlags = (leadlags...,)
    fargs = (ins..., map(x->Symbol(x.args...), leadlags)...)
    unknowns = _parseins((ins..., leadlags...))
    isempty(rawrets) && error("explicit return statement is not found")
    # If multiple return statements exist, only consider the first one
    outs, outexprs, tars, solver = _parsereturnimplicit(rawrets[1])
    tarexprs = _parseouts(tarnames[1])
    inits, calis = _parsevals(vals, outs)
    inexprs = _parseins(ntuple(i->calis[i][1], length(calis)))
    blkf = Symbol(f, :_block)
    return quote
        function $(esc(f))($(map(esc, fargs)...))
            $(esc(body))
        end
        function $(esc(blkf))()
            b = block($(esc(f)), $unknowns, $tarexprs; ($(map(esc, kwargs)...),)...)
            return block(b, $inexprs, $outexprs, $tarexprs,
                $(esc(calis)), $(esc(inits)), $(esc(tars));
                solver=$(esc(solver)), ($(map(esc, kwargs)...),)...)
        end
    end
end
