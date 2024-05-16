# Licenses

This directory contains the logic to extract and compile our license notice release document.

## All licenses

To get all of the licenses, run the following command:

  ```sh
  bazel build //tools/licenses:all_licenses
  ```

And copy the file output by the bazel build command into wherever you need it.

### Github API Key

To ensure that you don't get rate limited by Github, license builds need a Github API Key.
This is read from the GH_API_KEY env var.
