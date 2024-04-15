# Coding Style

The goal of our style guide is to create a uniform codebase and establish a canonical style. This allows for the code to be easier to read, understand and extend. To that end, these documents set to establish a set of best practices with the following intentions:

* Codify syle matters.
* Minimize surpises and issues during code review.
* Reduce debate on specific style items.

The intention is not to:

* Be an exhaustive list of comments for code review.
* List all of the rules everyone is expected to remember and follow at all times
* Replace good judgment in the use of language features and style.
* Justify large-scale changes to get rid of style differences

These style guides are intended to evolve over time. They provide guidelines to be followed for new or related code, but they do not justify making large-scale changes to existing code.

We aim to provide automation to provide enforcement of most of the style guide. Code reviewers should utilize judgement as the goal is not to nit-pick every aspect of a code review based on this style guide.


## Go

For Go we utilize the [Google style](https://google.github.io/styleguide/go) guide with the following exceptions, additions and callouts.

### Minimize interface sizes

Interfaces should be limited to create a proper abstraction "The bigger the interface, the weaker the abstraction."

For example:
```Go
// Interaces in general should end with -er. Prefer simple interfaces like.
type FleetGetter interface {
  FleetByID(string) (Fleet, error)
}

// Compared to:
type FleetManager interface {
  RenameFleetByID(string) (Fleet, error)
  FleetByID(string) (Fleet, error)
  DeleteFleetByID(string) error
}
```

### Testing

Refer to:

* [Useful Test Failures](https://google.github.io/styleguide/go/decisions#useful-test-failures)
* [Test Structure](https://google.github.io/styleguide/go/decisions#useful-test-failures)
* [Go Testing Patterns](https://github.com/gotestyourself/gotest.tools/wiki/Go-Testing-Patterns)

In general tests should be small and self contained. Over testing and poor abstractions lead to tests that are hard to maintain. If you see yourself consistently copying results from the terminal into tests, please rethink your testing strategy.

## C++

[Google Style Guide](https://google.github.io/styleguide/cppguide.html) with the following exceptions, callouts and additions:

### String Formatting

Use appropriate string formatting functions in different scenarios.

* Use std::string::operator+=() when you need to concatenate string.
* Use absl::StrJoin() to format a list of strings with separators.
* Use absl::StrFormat() when you actually need to **format string & non-string values**:
* absl::StrFormat("%0.4f", 1.0)
* Use absl::Substitute() when you need to **fill string & non-string values into a template**:
  * absl::Substitute("string $0, integer $1, float $2, class $3", "str", 1, 1.0, object);
* Use absl::StrCat() when you need to **concatenate string & non-string values**:
* Secondary: If performance is required prefer absl::StrCat()

**Example:**

  ```cpp
  absl::Substitute("[PID=$0] Unexpected event. Message: msg=$1.", pid, msg);
  ```

**Note:** The use of the standard << operator to create formatted strings is prohibited.

**Comments:** absl::Substitute is easy to use, and is slightly faster than absl::StrFormat() — even though we don’t typically use either in any performance critical code. Note that we previously used absl::StrFormat() in many places, but we should only use StrFormat() when it’s more powerful formatting is actually required.

### Integer types (sizes, signed/unsigned)

**Style:** If the variable is a count of or index into structures in memory, then always use size_t.

* This makes sure the count is appropriately sized for the addressable memory of the machine. Note that this rule covers containers, since they are countable objects in memory.

This rule does not apply to any countable value, however, as counts in time (e.g. stats counters) should not use size_t.

### emplace vs try_emplace

`try_emplace` is the newer version of `emplace`. Prefer `try_emplace` instead of `emplace` unless the object is not moveable. Note this is also the recommendation of absl.

## Typescript

[Google Style Guide](https://google.github.io/styleguide/tsguide.html) with teh following exceptions, callouts and additions:

## Python

[Google Style Guide](https://google.github.io/styleguide/pyguide.html) with teh following exceptions, callouts and additions:
