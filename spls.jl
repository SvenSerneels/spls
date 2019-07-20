@doc """
SPLS Univariate Sparse Partial Least squares Algorithms

Algorithm first outlined in:
    Sparse and robust PLS for binary classification,
    I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza,
    Journal of Chemometrics, 30 (2016), 153-162.

Parameters:
-----------
`eta`: float. Sparsity parameter in [0,1)

`n_components`: int, min 1. Note that if applied on data, n_components shall
    take a value <= minimum(size(x_data))

`verbose`: Bool: to print intermediate set of columns retained

`centre`: How to internally centre data. Accepts strings ("mean","median"),
    functions (e.g. Statistics.mean) or a location vector.

`scale`: How to internally scale data. Accepts strings ("std"),
        functions (e.g. Statistics.std) or a scale vector. Enter "none" both for
        centre and scale to work with the raw data.

`fit_algorithm`: Either "snipls", "sim_dense" or "sim_sparse".

`copy`: Bool, whether to copy data

Values:
-------
The mutable struct called by SNIPLS() will be populated with model results, such
    as coef_ (regression coefficients), x_scores_, etc., as well as estimated
    locations and scales for both X and y.

The module is consistent with the ScikitLearn API, e.g.

    import ScikitLearn.GridSearch:GridSearchCV
    gridsearch = GridSearchCV(snipls.SNIPLS(), Dict(:eta => collect(0.9:-0.01:0.1),
                    :n_components => collect(1:4), :verbose => false))
    ScikitLearn.fit!(gridsearch,X,y)
    gridsearch.best_params_
    ScikitLearn.predict(gridsearch.best_estimator_,Xt)


Written by Sven Serneels.

""" ->
module spls

    using Statistics, DataFrames, Parameters, Random
    import ScikitLearnBase: BaseRegressor, predict, fit!, @declare_hyperparameters
    include("C:\\Users\\SerneeS1\\OneDrive - BASF\\JulDocs\\SNIPLS\\_sreg_utils.jl")
    export spls, SPLS, autoscale

    @with_kw mutable struct SPLS <: BaseRegressor
        eta::Number = .5
        n_components::Int = 1
        verbose::Bool = true
        centre = "mean"
        scale = "none"
        copy::Bool = true
        fit_algorithm = "snipls"
        X_Types_Accept = [DataFrame,Array{Number,2},Array{Float64,2},
                        Array{Float32,2},Array{Int32,2},Array{Int64,2},
                        Array{Union{Missing, Float64},2}]
        y_Types_Accept = [DataFrame,Array{Number,1},Array{Float64,1},
                        Array{Float32,1},Array{Int32,1},Array{Int64,1},
                        Array{Union{Missing, Float64},1}]
        X0 = nothing
        y0 = nothing
        x_weights_ = nothing
        x_loadings_ = nothing
        C_ = nothing
        x_scores_ = nothing
        coef_ = nothing
        coef_scaled_ = nothing
        intercept_ = nothing
        x_ev_ = nothing
        y_ev_ = nothing
        fitted_ = nothing
        residuals_ = nothing
        x_Rweights_ = nothing
        colret_ = nothing
        x_loc_ = nothing
        y_loc_ = nothing
        x_sca_ = nothing
        y_sca_ = nothing
        x_names = nothing
        y_name = nothing
    end

    @declare_hyperparameters(SPLS, [:n_components, :eta, :centre, :scale, :verbose, :fit_algorithm])

    @doc """

        Dummy function equivalent to directly creating a SNIPLS struct

        """ ->
    function call(;kwargs...)

        self = SSIMPLS()
        if length(kwargs) > 0
            allkeys = keys(kwargs)
            for j in allkeys
                setfield!(self,j,kwargs[j])
            end #for
        end #if
        return(self)
    end #snipls

    @doc """

        Fit SNIPLS model to data X and y.

        """ ->
    function fit!(self,X,y)

        @assert self.fit_algorithm in ["snipls"] "Algorithm has to be `snipls`; `sim_dense`, `sim_sparse` still under construction]"

        if typeof(self.centre)==String
            if self.centre in ["mean","median"]
                self.centre = getfield(Statistics,Symbol(self.centre))
            else
                @assert self.centre=="none" "Only supported strings for median:" * "\n" *
                     "'mean', 'median', 'none'" * "\n" *
                     "Alternatively pass a function"
                # other location estimators can be included
            end
        end

        if typeof(self.scale)==String
            if self.scale in ["std"]
                self.scale = getfield(Statistics,Symbol(self.scale))
            else
                @assert self.scale=="none" "Only supported strings for scale:" * "\n" *
                     "'std','none'" * "\n" *
                     "Alternatively pass a function"
            end
        end

        X_Type = typeof(X)
        y_Type = typeof(y)
        @assert X_Type in self.X_Types_Accept "Supply X data as DataFrame or Matrix"
        @assert y_Type in self.y_Types_Accept "Supply y data as DataFrame or Vector"

        if self.copy
            setfield!(self,:X0,deepcopy(X))
            setfield!(self,:y0,deepcopy(y))
        end

        X0 = X
        y0 = y

        if y_Type == DataFrame
            ynames = true
            y_name = names(y0)
            y0 = y[:,1]
        else
            ynames = false
        end

        if X_Type == DataFrame
            Xnames = true
            X_name = names(X0)
            X0 = Matrix(X0)
        else
            Xnames = false
            X_name = nothing
        end

        (n,p) = size(X0)
        ny = size(y0,1)
        @assert ny == n "Number of cases in X and y needs to agree"

        centring_X = autoscale(X0,self.centre,self.scale)
        Xs= centring_X.X_as_
        mX = centring_X.col_loc_
        sX = centring_X.col_sca_

        centring_y = autoscale(y0,self.centre,self.scale)
        ys= centring_y.X_as_
        my = centring_y.col_loc_
        sy = centring_y.col_sca_
        nys = sum(ys.^2)
        s = Xs'*ys

        if self.fit_algorithm == "sim_dense"
            # (W,P,R,C,T,B,Xev,yev,colret) = _fit_dense(self,X,y,n,p,Xs,mX,sX,ys,my,sy,nys,Xnames,X_name,s)
            # experimental
        elseif self.fit_algorithm == "sim_sparse"
            # (W,P,R,C,T,B,Xev,yev,colret) = _fit_sparse(self,X,y,n,p,Xs,mX,sX,ys,my,sy,nys,Xnames,X_name,s)
            # experimental
        elseif self.fit_algorithm == "snipls"
            (W,P,R,C,T,B,Xev,yev,colret) = _fit_snipls(self,X,y,n,p,Xs,mX,sX,ys,my,sy,nys,Xnames,X_name,s)
        end


        B_rescaled = (sy./sX)' .* B
        yp_rescaled = X0*B_rescaled

        intercept = self.centre(y .- yp_rescaled)

        yfit = yp_rescaled .+ intercept
        r = y .- yfit
        setfield!(self,:x_weights_,W)
        setfield!(self,:x_Rweights_,R)
        setfield!(self,:x_loadings_,P)
        setfield!(self,:C_,C)
        setfield!(self,:x_scores_,T)
        setfield!(self,:coef_,B_rescaled)
        setfield!(self,:coef_scaled_,B)
        setfield!(self,:intercept_,intercept)
        setfield!(self,:x_ev_,Xev)
        setfield!(self,:y_ev_,yev)
        setfield!(self,:fitted_,yfit)
        setfield!(self,:residuals_,r)
        setfield!(self,:x_Rweights_,R)
        setfield!(self,:colret_,colret)
        setfield!(self,:x_loc_,mX)
        setfield!(self,:y_loc_,my)
        setfield!(self,:x_sca_,sX)
        setfield!(self,:y_sca_,sy)
        if Xnames
            setfield!(self,:x_names,X_name)
        end
        if ynames
            setfield!(self,:y_name,y_name)
        end

        return(self)

    end

    function _fit_snipls(self,X0,y0,n,p,Xs,mX,sX,ys,my,sy,nys,Xnames,X_name,wh=[])

        T = zeros(n,self.n_components)
        W = zeros(p,self.n_components)
        P = zeros(p,self.n_components)
        C = zeros(self.n_components,1)
        Xev = zeros(self.n_components,1)
        yev = zeros(self.n_components,1)
        B = zeros(p,1)
        oldgoodies = []
        Xi = Xs #deepcopy ?
        yi = ys

        if length(wh) == 0
            wh = Xs'*ys
        end

        for i in 1:self.n_components
            if i > 1
                wh =  Xi' * yi
            end
            (wh, goodies) =  _find_sparse(wh,self.eta)
            global goodies = union(oldgoodies,goodies)
            oldgoodies = goodies
            if length(goodies)==0
                print("No variables retained at" * string(i) * "latent variables" *
                      "and lambda = " * string(self.eta) * ", try lower lambda")
                break
            end
            elimvars = setdiff(1:p,goodies)
            wh[elimvars] .= 0
            th = Xi * wh
            nth2 = sum(th.^2)
            ch = (yi'*th) / nth2
            ph = (Xi' * Xi * wh) / nth2
            ph[elimvars] .= 0
            yi -= th*ch
            W[:,i] = wh
            P[:,i] = ph
            C[i] = ch
            T[:,i] = th
            Xi -= th * ph'
            Xev[i] = 100 - sum((Xs - T[:,1:i]*P[:,1:i]').^2) / sum(Xs.^2)*100
            yev[i] = 100 - sum((ys - T[:,1:i]*C[1:i]).^2) / nys *100
            if Xnames
                global colret = X_name[setdiff((1:p),elimvars)]
            else
                global colret = goodies
            end
            if self.verbose
                print("Variables retained for " * string(i) *
                        " latent variable(s):" *  "\n" * string(colret) * ".\n")
            end
        end

        if length(goodies) > 0
            R = W * inv(P'*W)
            all0 = findall(P.==0)
            R[all0] .= 0
            B = R * C
        else
            B = zeros(p,1)
            R = B
            T = zeros(n,self.n_components)
        end

        return((W,P,R,C,T,B,Xev,yev,colret))

    end



    @doc """

        Fit SNIPLS model to data X and y and only return the regression
        coefficients.

        """ ->
    function fit(self,X,y)

        if self.coef_ == nothing
            fit!(self,X,y)
        end

        return(self.coef_)

    end

    @doc """

        Predict responses for new predictor data.

        """ ->
    function predict(self,Xn)

        Xn_type = typeof(Xn)
        @assert Xn_type in self.X_Types_Accept "Supply new X data as DataFrame or Matrix"
        if Xn_type == DataFrame
            Xn = Matrix(Xn)
        end
        (n,p) = size(Xn)
        @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
        return((Xn * self.coef_ .+ self.intercept_)[:,1])

    end

    @doc """

        Transform new predictor data to estimated scores.

        """ ->
    function transform(self,Xn)

        Xn_type = typeof(Xn)
        @assert Xn_type in self.X_Types_Accept "Supply new X data as DataFrame or Matrix"
        if Xn_type == DataFrame
            Xn = Matrix(Xn)
        end
        (n,p) = size(Xn)
        @assert p==length(self.coef_) "New data must have same number of columns as the ones the model has been trained with"
        Xnc = autoscale(Xn,self.x_loc_,self.x_sca_)
        return(Xnc*self.x_Rweights_)

    end

    @doc """

        Get all settings from an existing SNIPLS struct (also the ones not
            declared to ScikitLearn)

        """ ->
    function get_all_params(self::T, param_names=[]::Array{Any}) where{T}

        if length(param_names)==0

            params_dict = type2dict(self)

        else

            params_dict = Dict{Symbol, Any}()

            for name::Symbol in param_names
                params_dict[name] = getfield(self, name)
            end

        end

        params_dict

    end

    @doc """

        ScikitLearn similar function to set parameters in an existing SNIPLS
            struct, yet takes a Dict as an argument.

        Compare:
        ScikitLearn.set_params!(lnrj,eta=.3)
        snipls.set_params_dict!(lnrj,Dict(:eta => .3))

        """ ->
    function set_params_dict!(self::T, params; param_names=nothing) where {T}

        for (k, v) in params

            if param_names !== nothing && !(k in param_names)

                throw(ArgumentError("An estimator of type $T was passed the invalid hyper-parameter $k. Valid hyper-parameters: $param_names"))

            end

            setfield!(self, k, v)

        end

        print(self)

    end

    clone_param(v::Any) = v # fall-back

    clone_param(rng::Random.MersenneTwister) = deepcopy(rng) # issue #15698. Solved in 0.5

    function is_classifier(self) return(false) end

    @doc """

        ScikitLearn compatible function to clone an existing SNIPLS struct.

        """ ->
    function clone(self::T) where {T}

        kw_params = Dict{Symbol, Any}()

        # cloning the values is scikit-learn's default behaviour. It's ok?

        for (k, v) in get_params(self) kw_params[k] = clone_param(v) end

        return T(; kw_params...)

    end

end
