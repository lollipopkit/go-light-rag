package handler

import golightrag "github.com/lollipopkit/go-light-rag"

//nolint:lll
const defaultEntityExtractionGoal = `
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.`

//nolint:lll
const defaultKeywordExtractionGoal = `
Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.`

var defaultEntityTypes = []string{"organization", "person", "geo", "event", "category"}

//nolint:lll
var defaultEntityExtractionExamples = []golightrag.EntityExtractionPromptExample{
	{
		EntityTypes: []string{"person", "technology", "mission", "organization", "location"},
		Text: `
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "Alex",
				Type:        "person",
				Description: "Alex is a character who experiences frustration and is observant of the dynamics among other characters.",
			},
			{
				Name:        "Taylor",
				Type:        "person",
				Description: "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.",
			},
			{
				Name:        "Jordan",
				Type:        "person",
				Description: "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.",
			},
			{
				Name:        "Cruz",
				Type:        "person",
				Description: "Cruz is associated with a vision of control and order, influencing the dynamics among other characters.",
			},
			{
				Name:        "The Device",
				Type:        "technology",
				Description: "The Device is central to the story, with potential game-changing implications, and is revered by Taylor.",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "Alex",
				TargetEntity: "Taylor",
				Description:  "Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device.",
				Keywords:     []string{"power dynamics", "perspective shift"},
				Strength:     7,
			},
			{
				SourceEntity: "Alex",
				TargetEntity: "Jordan",
				Description:  "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.",
				Keywords:     []string{"shared goals", "rebellion"},
				Strength:     6,
			},
			{
				SourceEntity: "Taylor",
				TargetEntity: "Jordan",
				Description:  "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.",
				Keywords:     []string{"conflict resolution", "mutual respect"},
				Strength:     8,
			},
			{
				SourceEntity: "Jordan",
				TargetEntity: "Cruz",
				Description:  "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.",
				Keywords:     []string{"ideological conflict", "rebellion"},
				Strength:     5,
			},
			{
				SourceEntity: "Taylor",
				TargetEntity: "The Device",
				Description:  "Taylor shows reverence towards the device, indicating its importance and potential impact.",
				Keywords:     []string{"reverence", "technological significance"},
				Strength:     9,
			},
		},
	},
	{
		EntityTypes: []string{"company", "index", "commodity", "market_trend", "economic_policy", "biological"},
		Text: `
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve’s next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "Global Tech Index",
				Type:        "index",
				Description: "The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today.",
			},
			{
				Name:        "Nexon Technologies",
				Type:        "company",
				Description: "Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.",
			},
			{
				Name:        "Omega Energy",
				Type:        "company",
				Description: "Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.",
			},
			{
				Name:        "Gold Futures",
				Type:        "commodity",
				Description: "Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.",
			},
			{
				Name:        "Crude Oil",
				Type:        "commodity",
				Description: "Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand.",
			},
			{
				Name:        "Market Selloff",
				Type:        "market_trend",
				Description: "Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations.",
			},
			{
				Name:        "Federal Reserve Policy Announcement",
				Type:        "economic_policy",
				Description: "The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability.",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "Global Tech Index",
				TargetEntity: "Market Selloff",
				Description:  "The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns.",
				Keywords:     []string{"market performance", "investor sentiment"},
				Strength:     9,
			},
			{
				SourceEntity: "Nexon Technologies",
				TargetEntity: "Global Tech Index",
				Description:  "Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index.",
				Keywords:     []string{"company impact", "index movement"},
				Strength:     8,
			},
			{
				SourceEntity: "Gold Futures",
				TargetEntity: "Market Selloff",
				Description:  "Gold prices rose as investors sought safe-haven assets during the market selloff.",
				Keywords:     []string{"market reaction", "safe-haven investment"},
				Strength:     10,
			},
			{
				SourceEntity: "Federal Reserve Policy Announcement",
				TargetEntity: "Market Selloff",
				Description:  "Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff.",
				Keywords:     []string{"interest rate impact", "financial regulation"},
				Strength:     7,
			},
		},
	},
	{
		EntityTypes: []string{"economic_policy", "athlete", "event", "location", "record", "organization", "equipment"},
		Text: `
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
    `,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "World Athletics Championship",
				Type:        "event",
				Description: "The World Athletics Championship is a global sports competition featuring top athletes in track and field.",
			},
			{
				Name:        "Tokyo",
				Type:        "location",
				Description: "Tokyo is the host city of the World Athletics Championship.",
			},
			{
				Name:        "Noah Carter",
				Type:        "athlete",
				Description: "Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship.",
			},
			{
				Name:        "100m Sprint Record",
				Type:        "record",
				Description: "The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter.",
			},
			{
				Name:        "Carbon-Fiber Spikes",
				Type:        "equipment",
				Description: "Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction.",
			},
			{
				Name:        "World Athletics Federation",
				Type:        "organization",
				Description: "The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations.",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "World Athletics Championship",
				TargetEntity: "Tokyo",
				Description:  "The World Athletics Championship is being hosted in Tokyo.",
				Keywords:     []string{"event location", "international competition"},
				Strength:     8,
			},
			{
				SourceEntity: "Noah Carter",
				TargetEntity: "100m Sprint Record",
				Description:  "Noah Carter set a new 100m sprint record at the championship.",
				Keywords:     []string{"athlete achievement", "record-breaking"},
				Strength:     10,
			},
			{
				SourceEntity: "Noah Carter",
				TargetEntity: "Carbon-Fiber Spikes",
				Description:  "Noah Carter used carbon-fiber spikes to enhance performance during the race.",
				Keywords:     []string{"athletic equipment", "performance boost"},
				Strength:     7,
			},
			{
				SourceEntity: "World Athletics Federation",
				TargetEntity: "100m Sprint Record",
				Description:  "The World Athletics Federation is responsible for validating and recognizing new sprint records.",
				Keywords:     []string{"sports regulation", "record certification"},
				Strength:     9,
			},
		},
	},
}

//nolint:lll
var defaultKeywordExtractionExamples = []golightrag.KeywordExtractionPromptExample{
	{
		Query:             "How does international trade influence global economic stability?",
		LowLevelKeywords:  []string{"Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"},
		HighLevelKeywords: []string{"International trade", "Global economic stability", "Economic impact"},
	},
	{
		Query:             "What are the environmental consequences of deforestation on biodiversity?",
		LowLevelKeywords:  []string{"Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"},
		HighLevelKeywords: []string{"Environmental consequences", "Deforestation", "Biodiversity loss"},
	},
	{
		Query:             "What is the role of education in reducing poverty?",
		LowLevelKeywords:  []string{"School access", "Literacy rates", "Job training", "Income inequality"},
		HighLevelKeywords: []string{"Education", "Poverty reduction", "Socioeconomic development"},
	},
}

//nolint:lll
const goEntityExtractionGoal = `
Given a Go code document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the code and all relationships among the identified entities. 

Note that each code chunk will contain exactly one primary entity (a function, struct, interface, method, etc.) along with its package declaration for context. The chunk may reference other entities that are defined elsewhere in the codebase. Pay special attention to:

- The package declaration that always appears at the top of each chunk
- Documentation comments that describe entity purpose and behavior
- References to other entities that may be defined in other chunks (e.g., global variables, constants, types, functions)
- Method receivers that indicate a relationship with a struct or interface
- Imported packages and their usage
- Implicit relationships through variable usage, function calls, or type references

For referenced identifiers where you cannot determine if they are constants or variables:
- Extract them as both "const" AND "variable" entity types
- In the description, note that this entity is referenced but not defined in the current chunk
- The ambiguity will be resolved when analyzing the chunk where the entity is defined

Extract both the defined entity in the chunk and any referenced entities, even if you only see their usage and not their definition. Identify all relationships between entities, including those that span across different code chunks.`

//nolint:lll
const goKeywordExtractionGoal = `
Given queries and conversation history related to Go codebases, extract both high-level and low-level keywords that would be relevant for finding appropriate code chunks in a RAG system.

High-level keywords should focus on architectural concepts, patterns, and design principles specific to the codebase being queried.

Low-level keywords should focus on specific package names, type names, function names, method names, and implementation details that would help locate the precise code chunks relevant to the query.

The keywords should help retrieve the most contextually appropriate code chunks from the codebase to answer specific questions about implementation details, usage patterns, or architectural decisions.`

var goEntityTypes = []string{"package", "function", "method", "struct", "interface", "const", "variable", "import"}

//nolint:lll
var goEntityExtractionExamples = []golightrag.EntityExtractionPromptExample{
	{
		EntityTypes: goEntityTypes,
		Text: `package calculator

import (
	"fmt"
	"math"
)`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "calculator",
				Type:        "package",
				Description: "A package that provides calculator functionality",
			},
			{
				Name:        "fmt",
				Type:        "import",
				Description: "An imported package for formatted I/O operations",
			},
			{
				Name:        "math",
				Type:        "import",
				Description: "An imported package for mathematical operations and constants",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "calculator",
				TargetEntity: "fmt",
				Description:  "The calculator package imports the fmt package",
				Keywords:     []string{"import dependency", "package relationship"},
				Strength:     7,
			},
			{
				SourceEntity: "calculator",
				TargetEntity: "math",
				Description:  "The calculator package imports the math package",
				Keywords:     []string{"import dependency", "package relationship"},
				Strength:     7,
			},
		},
	},
	{
		EntityTypes: goEntityTypes,
		Text: `package calculator

// Calculator represents a simple calculator with memory
type Calculator struct {
	Memory float64
}`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "calculator",
				Type:        "package",
				Description: "A package that provides calculator functionality",
			},
			{
				Name:        "Calculator",
				Type:        "struct",
				Description: "A struct representing a calculator with memory storage capability",
			},
			{
				Name:        "Memory",
				Type:        "variable",
				Description: "A field in the Calculator struct that stores a float64 value",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "Calculator",
				TargetEntity: "calculator",
				Description:  "Calculator struct is defined in the calculator package",
				Keywords:     []string{"struct definition", "package member"},
				Strength:     9,
			},
			{
				SourceEntity: "Memory",
				TargetEntity: "Calculator",
				Description:  "Memory is a field within the Calculator struct",
				Keywords:     []string{"struct field", "data storage"},
				Strength:     10,
			},
		},
	},
	{
		EntityTypes: goEntityTypes,
		Text: `package calculator

// Add adds two numbers and returns the result
func Add(a, b float64) float64 {
	return a + b
}`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "calculator",
				Type:        "package",
				Description: "A package that provides calculator functionality",
			},
			{
				Name:        "Add",
				Type:        "function",
				Description: "A function that takes two float64 parameters and returns their sum",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "Add",
				TargetEntity: "calculator",
				Description:  "Add function is defined in the calculator package",
				Keywords:     []string{"function definition", "package member"},
				Strength:     9,
			},
		},
	},
	{
		EntityTypes: goEntityTypes,
		Text: `package calculator

// SaveToMemory stores a result in the calculator's memory
func (c *Calculator) SaveToMemory(value float64) {
	c.Memory = value
}`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "calculator",
				Type:        "package",
				Description: "A package that provides calculator functionality",
			},
			{
				Name:        "SaveToMemory",
				Type:        "method",
				Description: "A method on Calculator that stores a value in the calculator's memory",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "SaveToMemory",
				TargetEntity: "calculator",
				Description:  "SaveToMemory method is defined in the calculator package",
				Keywords:     []string{"method definition", "package member"},
				Strength:     8,
			},
			{
				SourceEntity: "SaveToMemory",
				TargetEntity: "Calculator",
				Description:  "SaveToMemory is a method on the Calculator struct that modifies its state",
				Keywords:     []string{"method implementation", "state modification"},
				Strength:     10,
			},
		},
	},
	{
		EntityTypes: goEntityTypes,
		Text: `package shapes

// Shape defines methods that all shapes must implement
type Shape interface {
	Area() float64
	Perimeter() float64
}`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "shapes",
				Type:        "package",
				Description: "A package containing shape-related types and calculations",
			},
			{
				Name:        "Shape",
				Type:        "interface",
				Description: "An interface that defines methods all shapes must implement",
			},
			{
				Name:        "Area",
				Type:        "method",
				Description: "A method signature required by the Shape interface",
			},
			{
				Name:        "Perimeter",
				Type:        "method",
				Description: "A method signature required by the Shape interface",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "Shape",
				TargetEntity: "shapes",
				Description:  "Shape interface is defined in the shapes package",
				Keywords:     []string{"interface definition", "package member"},
				Strength:     9,
			},
			{
				SourceEntity: "Area",
				TargetEntity: "Shape",
				Description:  "Area is a method required by the Shape interface",
				Keywords:     []string{"interface method", "contract requirement"},
				Strength:     10,
			},
			{
				SourceEntity: "Perimeter",
				TargetEntity: "Shape",
				Description:  "Perimeter is a method required by the Shape interface",
				Keywords:     []string{"interface method", "contract requirement"},
				Strength:     10,
			},
		},
	},
	{
		EntityTypes: goEntityTypes,
		Text: `package config

import "time"

// Default timeout values for application
const (
	DefaultTimeout = 30 * time.Second
	MaxRetries     = 3
)`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "config",
				Type:        "package",
				Description: "A package providing configuration functionality for an application",
			},
			{
				Name:        "DefaultTimeout",
				Type:        "const",
				Description: "A constant defining the default timeout duration for the application",
			},
			{
				Name:        "MaxRetries",
				Type:        "const",
				Description: "A constant defining the maximum number of retry attempts",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "DefaultTimeout",
				TargetEntity: "config",
				Description:  "DefaultTimeout constant is defined in the config package",
				Keywords:     []string{"constant definition", "package member"},
				Strength:     8,
			},
			{
				SourceEntity: "MaxRetries",
				TargetEntity: "config",
				Description:  "MaxRetries constant is defined in the config package",
				Keywords:     []string{"constant definition", "package member"},
				Strength:     8,
			},
		},
	},
	{
		EntityTypes: goEntityTypes,
		Text: `package handler

// EntityExtractionPromptData returns the data needed to generate prompts for extracting
// entities and relationships from text content.
func (d Default) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
    language := d.Language
    if language == "" {
        language = defaultLanguage
    }
    examples := d.EntityExtractionExamples
    if examples == nil {
        examples = defaultEntityExtractionExamples
    }
    return golightrag.EntityExtractionPromptData{
        Goal:        goEntityExtractionGoal,
        EntityTypes: goEntityTypes,
        Language:    language,
        Examples:    examples,
    }
}`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "handler",
				Type:        "package",
				Description: "A package providing handler functionality for RAG operations",
			},
			{
				Name:        "EntityExtractionPromptData",
				Type:        "method",
				Description: "A method on Default struct that returns data needed for entity extraction prompts",
			},
			{
				Name:        "Default",
				Type:        "struct",
				Description: "A struct that this method belongs to, defined elsewhere in the codebase",
			},
			{
				Name:        "defaultLanguage",
				Type:        "const",
				Description: "A constant containing the default language setting, referenced but not defined in this chunk",
			},
			{
				Name:        "defaultLanguage",
				Type:        "variable",
				Description: "A variable containing the default language setting, referenced but not defined in this chunk",
			},
			{
				Name:        "defaultEntityExtractionExamples",
				Type:        "const",
				Description: "A constant containing default examples for entity extraction, referenced but not defined in this chunk",
			},
			{
				Name:        "defaultEntityExtractionExamples",
				Type:        "variable",
				Description: "A variable containing default examples for entity extraction, referenced but not defined in this chunk",
			},
			{
				Name:        "goEntityExtractionGoal",
				Type:        "const",
				Description: "A constant containing the goal statement for Go entity extraction, referenced but not defined in this chunk",
			},
			{
				Name:        "goEntityExtractionGoal",
				Type:        "variable",
				Description: "A variable containing the goal statement for Go entity extraction, referenced but not defined in this chunk",
			},
			{
				Name:        "goEntityTypes",
				Type:        "const",
				Description: "A constant containing entity types for Go code analysis, referenced but not defined in this chunk",
			},
			{
				Name:        "goEntityTypes",
				Type:        "variable",
				Description: "A variable containing entity types for Go code analysis, referenced but not defined in this chunk",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "EntityExtractionPromptData",
				TargetEntity: "handler",
				Description:  "EntityExtractionPromptData method is defined in the handler package",
				Keywords:     []string{"method definition", "package member"},
				Strength:     8,
			},
			{
				SourceEntity: "EntityExtractionPromptData",
				TargetEntity: "Default",
				Description:  "EntityExtractionPromptData is a method on the Default struct",
				Keywords:     []string{"method receiver", "struct method"},
				Strength:     10,
			},
			{
				SourceEntity: "EntityExtractionPromptData",
				TargetEntity: "defaultLanguage",
				Description:  "EntityExtractionPromptData references defaultLanguage as a fallback value",
				Keywords:     []string{"variable usage", "default value"},
				Strength:     7,
			},
			{
				SourceEntity: "EntityExtractionPromptData",
				TargetEntity: "defaultEntityExtractionExamples",
				Description:  "EntityExtractionPromptData references defaultEntityExtractionExamples as a fallback value",
				Keywords:     []string{"variable usage", "default value"},
				Strength:     7,
			},
			{
				SourceEntity: "EntityExtractionPromptData",
				TargetEntity: "goEntityExtractionGoal",
				Description:  "EntityExtractionPromptData uses goEntityExtractionGoal in the returned data structure",
				Keywords:     []string{"variable usage", "return value"},
				Strength:     9,
			},
			{
				SourceEntity: "EntityExtractionPromptData",
				TargetEntity: "goEntityTypes",
				Description:  "EntityExtractionPromptData uses goEntityTypes in the returned data structure",
				Keywords:     []string{"variable usage", "return value"},
				Strength:     9,
			},
		},
	},
}

//nolint:lll
var goKeywordExtractionExamples = []golightrag.KeywordExtractionPromptExample{
	{
		Query:             "How does the SSEClient maintain connection with the server and handle reconnection?",
		LowLevelKeywords:  []string{"SSEClient", "Reconnect", "ConnectWithRetry", "EventSource", "readLoop", "LastEventID", "backoff", "http.Client"},
		HighLevelKeywords: []string{"connection maintenance", "retry logic", "event streaming", "persistent connections"},
	},
	{
		Query:             "What's the event delivery guarantee mechanism in the SSE server implementation?",
		LowLevelKeywords:  []string{"SSEServer", "Broadcast", "Subscribe", "eventStore", "EventID", "clientConnections", "mutex", "channel"},
		HighLevelKeywords: []string{"message delivery", "event buffering", "concurrency control", "broadcast pattern"},
	},
	{
		Query:             "How does the transaction isolation work in this key-value database?",
		LowLevelKeywords:  []string{"DB.Begin", "Tx", "Commit", "Rollback", "writable", "rwlock", "meta", "page", "mmap"},
		HighLevelKeywords: []string{"transaction isolation", "MVCC", "B+tree", "durability guarantees"},
	},
	{
		Query:             "What's the bucket structure for organizing data in the key-value store?",
		LowLevelKeywords:  []string{"Bucket", "CreateBucket", "CreateBucketIfNotExists", "NextSequence", "Cursor", "bucket.Put", "bucket.Get", "nested buckets"},
		HighLevelKeywords: []string{"data organization", "key hierarchy", "bucket pattern", "namespaces"},
	},
	{
		Query:             "How does middleware chaining work in this HTTP framework?",
		LowLevelKeywords:  []string{"gin.HandlerFunc", "gin.Engine", "RouterGroup", "Use", "Next", "Abort", "c.Request", "c.Writer"},
		HighLevelKeywords: []string{"middleware chain", "request pipeline", "context propagation", "handler composition"},
	},
	{
		Query:             "What's the best way to implement custom validation for request parameters?",
		LowLevelKeywords:  []string{"validator", "ShouldBindJSON", "ShouldBindQuery", "BindJSON", "binding.Validator", "gin.Context", "validation.RegisterValidation"},
		HighLevelKeywords: []string{"request validation", "custom validators", "binding middleware", "input sanitization"},
	},
	{
		Query:             "How does the controller reconciliation loop handle errors in Kubernetes?",
		LowLevelKeywords:  []string{"Reconcile", "controller.Result", "requeueAfter", "client.Get", "client.Update", "apierrors.IsNotFound", "ctrl.Log", "manager.GetClient"},
		HighLevelKeywords: []string{"reconciliation pattern", "error handling", "control loop", "eventual consistency"},
	},
	{
		Query:             "What's the mechanism for leader election in Kubernetes controllers?",
		LowLevelKeywords:  []string{"leaderelection", "resourcelock", "LeaseLock", "LeaderElectionConfig", "OnStartedLeading", "OnStoppedLeading", "NewLeaderElector", "LeaseDurationSeconds"},
		HighLevelKeywords: []string{"leader election", "distributed coordination", "controller redundancy", "high availability"},
	},
}

//nolint:lll
const semanticChunkingPrompt = `---Goal---
Analyze the given text and divide it into semantically coherent sections based on thematic shifts, logical structure, or natural section boundaries.

---Instructions---
1. Carefully read and understand the entire text.
2. Identify natural semantic boundaries where the topic, focus, or theme shifts.
3. Divide the text into non-overlapping sections at these boundaries.
4. For each section, provide:
   - section_summary: A brief summary of the section's content (2-3 sentences)
   - start_position: The position in the text where this section begins (use character index, starting from 0)
   - end_position: The position in the text where this section ends (use character index)

5. Format your output as a VALID JSON object with the following structure:
{
  "sections": [
    {
      "section_summary": string,
      "start_position": number,
      "end_position": number
    }
  ]
}

6. The JSON output MUST be valid JSON with no explanation text before or after it. Do not include any markdown formatting like backticks, and do not include any text outside the JSON structure.

######################
---Examples---
######################
Example 1:

Text:
The history of computing spans thousands of years. Early computing tools included the abacus and mechanical calculators. The first electronic computers were created in the 1940s, with ENIAC being one of the earliest general-purpose computers. The development of the transistor in the 1950s led to smaller, more reliable computers. In the 1960s and 1970s, mainframe computers dominated the industry, with companies like IBM leading the way. The personal computer revolution began in the late 1970s with the introduction of microcomputers like the Apple II and later the IBM PC. The 1990s saw the rise of the internet and the World Wide Web, transforming how people use computers and access information. The 21st century has been characterized by mobile computing, cloud services, and advances in artificial intelligence.

################
Output:
{
  "sections": [
    {
      "section_summary": "Early computing history, from ancient calculation tools to the first electronic computers of the 1940s.",
      "start_position": 0,
      "end_position": 180
    },
    {
      "section_summary": "Development of computing from the 1950s through the 1970s, focusing on transistors and mainframe computers.",
      "start_position": 181,
      "end_position": 342
    },
    {
      "section_summary": "The personal computer revolution beginning in the late 1970s.",
      "start_position": 343,
      "end_position": 467
    },
    {
      "section_summary": "The internet era of the 1990s and the modern computing landscape of the 21st century.",
      "start_position": 468,
      "end_position": 646
    }
  ]
}

Example 2:

Text:
# Introduction to Python

Python is a high-level, interpreted programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991.

## Key Features

Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.

Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, and functional programming.

## Applications

Python is widely used in many different fields, such as:

- Web development (with frameworks like Django and Flask)
- Data analysis and visualization (with libraries like pandas and matplotlib)
- Machine learning and artificial intelligence (with libraries like TensorFlow and PyTorch)
- Scientific computing (with libraries like NumPy and SciPy)
- Automation and scripting

## Getting Started

To get started with Python, you'll need to install the Python interpreter on your computer. You can download it from the official website: https://www.python.org/downloads/

Once installed, you can run Python code interactively in the Python shell or by creating and running .py files.

################
Output:
{
  "sections": [
    {
      "section_summary": "Introduction to Python, its creation by Guido van Rossum, and initial release in 1991.",
      "start_position": 0,
      "end_position": 135
    },
    {
      "section_summary": "Key features of Python including its design philosophy, typing system, and supported programming paradigms.",
      "start_position": 136,
      "end_position": 412
    },
    {
      "section_summary": "Various applications and fields where Python is commonly used, including web development, data analysis, machine learning, scientific computing, and automation.",
      "start_position": 413,
      "end_position": 701
    },
    {
      "section_summary": "Instructions for getting started with Python, including installation and running Python code.",
      "start_position": 702,
      "end_position": 901
    }
  ]
}

#############################
---Real Data---
######################
Text:
{{.Content}}
######################
Output:`
