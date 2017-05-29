
function createModel(Dsize, nout, KKw, shared, div)
    	-- define model to train
     	local featext = nn.Sequential()
    	local D     = Dsize 
    	local kW    = KKw 
    	local dW    = 1 
    	local NumFilter = D/div
    	local sepModel = shared
 
    --	dofile "PaddingReshape.lua"
		
		deepQuery=nn.Sequential()
   		D = Dsize 
		local incep1max = nn.Sequential()
		incep1max:add(nn.TemporalConvolution(D,NumFilter,1,dw))
		incep1max:add(nn.Tanh())
		incep1max:add(nn.Max(1))
		incep1max:add(nn.Reshape(NumFilter,1))		  
--		local incep2max = nn.Sequential()
--		incep2max:add(nn.Max(1))
--		incep2max:add(nn.View(torch.DoubleTensor{1,D}))
--        	incep2max:add(nn.TemporalConvolution(D,NumFilter,1,dw))
--		incep2max:add(nn.Tanh())
--		incep2max:add(nn.Reshape(NumFilter,1))			  
		local combineDepth = nn.Concat(2)
		combineDepth:add(incep1max)
--		combineDepth:add(incep2max)
		  
		local ngram = kW                
		for cc = 2, ngram do
		    local incepMax = nn.Sequential()
		    incepMax:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
		    incepMax:add(nn.Tanh())
		    incepMax:add(nn.Max(1))
		    incepMax:add(nn.Reshape(NumFilter,1))		    		    
		    combineDepth:add(incepMax)		    
		end  		  
		  	
		local incep1mean = nn.Sequential()
		incep1mean:add(nn.TemporalConvolution(D,NumFilter,1,dw))
		incep1mean:add(nn.Tanh())
		incep1mean:add(nn.Mean(1))
		incep1mean:add(nn.Reshape(NumFilter,1))		    		  		  
--		local incep2mean = nn.Sequential()
--		incep2mean:add(nn.Mean(1))
--		incep2mean:add(nn.View(torch.DoubleTensor{1,D}))
--		incep2mean:add(nn.TemporalConvolution(D,NumFilter,1,dw))
--		incep2mean:add(nn.Tanh())
--		incep2mean:add(nn.Reshape(NumFilter,1))		  
		combineDepth:add(incep1mean)
--		combineDepth:add(incep2mean)		  
		for cc = 2, ngram do
		    local incepMean = nn.Sequential()
		    incepMean:add(nn.TemporalConvolution(D,NumFilter,cc,dw))
		    incepMean:add(nn.Tanh())
		    incepMean:add(nn.Mean(1))
		    incepMean:add(nn.Reshape(NumFilter,1))			    
		    combineDepth:add(incepMean)	
		end  
		  
		featext:add(combineDepth)		
		if sepModel == 1 then  
			modelQ= featext:clone('weight','bias','gradWeight','gradBias')
		else			
			modelQ= featext:clone()
		end
		paraQuery=nn.ParallelTable()
		paraQuery:add(modelQ)
       	paraQuery:add(featext)			
       	
       	deepQuery:add(paraQuery) 
		deepQuery:add(nn.JoinTable(2)) 
			
		d=nn.Concat(1) 
        
        local MaxMean = 2
		local items = (ngram)*MaxMean
		--local separator = items
			 				
		for i=1,NumFilter do
			for j=1,2 do
				local connection = nn.Sequential()
				connection:add(nn.Select(1,i)) -- == 2items
				connection:add(nn.Reshape(2*items,1)) --2items*1 here					
				local minus=nn.Concat(2)
				local c1=nn.Sequential()
				local c2=nn.Sequential()
				if j == 1 then 
					c1:add(nn.Narrow(1,1,ngram)) -- first half (items/2)
					c2:add(nn.Narrow(1,items+1,ngram)) -- first half (items/2)
				else 
					c1:add(nn.Narrow(1,ngram+1,ngram)) -- 
					c2:add(nn.Narrow(1,items+ngram+1,ngram)) --each is ngram+1 portion (max or mean)
				end						
				minus:add(c1)
				minus:add(c2)
				connection:add(minus) 				
				local similarityC=nn.Concat(1) 	
--				local s1=nn.Sequential()
--				s1:add(nn.SplitTable(2))
--				s1:add(nn.PairwiseDistance(2)) -- scalar
				local s2=nn.Sequential()					
				s2:add(nn.SplitTable(2)) 
				s2:add(nn.CsDis()) -- scalar
--				local s3=nn.Sequential()
--				s3:add(nn.SplitTable(2))
--				s3:add(nn.CSubTable()) -- linear
--				s3:add(nn.Abs())
--				similarityC:add(s1)
				similarityC:add(s2)	
--				similarityC:add(s3)				
				connection:add(similarityC)										
				d:add(connection)				
			end
		end			
		
    	deepQuery:add(d)	    
    	return deepQuery	
end


