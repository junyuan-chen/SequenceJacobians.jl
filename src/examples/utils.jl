module ExampleUtils

export lhs_equals_rhs_interpolate!

function lhs_equals_rhs_interpolate!(li, lp, lhs, rhs, imax=size(rhs, 1), jmax=size(rhs, 2))
    i = 1
    @inbounds for j in 1:jmax
        while true
            if lhs[i] < rhs[i,j]
                break
            elseif i < imax
                i += 1
            else
                break
            end
        end
        if i == 1
            li[j] = 1
            lp[j] = 1.0
        else
            li[j] = i - 1
            err_upper = rhs[i,j] - lhs[i]
            err_lower = rhs[i-1,j] - lhs[i-1]
            lp[j] = err_upper / (err_upper-err_lower)
        end
    end
end

end
